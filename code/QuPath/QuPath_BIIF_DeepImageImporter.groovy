import ij.measure.Calibration
import ij.plugin.filter.ThresholdToSelection
import ij.process.ByteProcessor
import ij.process.ImageProcessor
import qupath.imagej.tools.IJTools
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.classes.PathClassFactory
import static qupath.lib.gui.scripting.QPEx.*

import javax.imageio.ImageIO
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects


def loadMask(hierarchy, pngFile, className) {
    def img = ImageIO.read(pngfile)
    // To create the ROI, travel into ImageJ
    def bp = new ByteProcessor(img)
    bp.setThreshold(127.5, Double.MAX_VALUE, ImageProcessor.NO_LUT_UPDATE)
    def roiIJ = new ThresholdToSelection().convert(bp)
    
    int x = 0
    int y = 0
    int downsample = 8
    def plane = ImagePlane.getPlane(0, 0)
    
    // Convert ImageJ ROI to a QuPath ROI
    // This assumes we have a single 2D image (no z-stack, time series)
    // Currently, we need to create an ImageJ Calibration object to store the origin
    // (this might be simplified in a later version)
    def cal = new Calibration()
    cal.xOrigin = -x/downsample
    cal.yOrigin = -y/downsample
    def roi = IJTools.convertToROI(roiIJ, cal, downsample,plane)
    
    // Create & return the object
    //fatAnnot = new PathAnnotationObject(roi, getPathClass(className))
    //def detectionPath = PathObjects.createDetectionObject(roi)
    //detectionPath.setPathClass(getPathClass(className))
    //addObject(detectionPath)

    def maskPath = new PathDetectionObject(roi, getPathClass(className))
    hierarchy.addPathObject(maskPath, true)
}

import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.io.GsonTools
import qupath.lib.gui.dialogs.Dialogs

//jsonfile = Dialogs.promptForFile("Choose json annotation file", null, "json",null)

String dir
def currentFile = new File(getCurrentServer().getURIs()[0])
dir = currentFile.getParentFile().getPath() + "/DL_Results/"
jsonfile = dir + "qupath_rois.json"
print (jsonfile);
def plane = ImagePlane.getPlane(0, 0)

def gson=GsonTools.getInstance(true)
BufferedReader bufferedReader = new BufferedReader(new FileReader(jsonfile));
HashMap<String, String> myjson = gson.fromJson(bufferedReader, HashMap.class); 

classes = [getPathClass('non-dilated'), getPathClass('int. dilated'), getPathClass('highly dilated')]

xCoords_crop = myjson["xCoords_crop"]
yCoords_crop = myjson["yCoords_crop"]

xCoords_vessel = myjson["xCoords_vessel"]
yCoords_vessel = myjson["yCoords_vessel"]
classes_index = myjson["classes"]

// Get main data structures
def imageData = getCurrentImageData()

for (int crop=0; crop < xCoords_crop.size(); crop++) {
    //example for creating an annotation selecting the "cropped" region
    def roi = ROIs.createPolygonROI(xCoords_crop[crop] as double[], yCoords_crop[crop] as double[], plane)
    def annotation = PathObjects.createAnnotationObject(roi)
    //addObject(annotation)
    imageData.getHierarchy().addPathObject(annotation, true)

    for (vessel=0; vessel < xCoords_vessel[crop].size(); vessel++) {
        //example for creating a vessel ROI and setting a classification

        def roi_vessel = ROIs.createPolygonROI(xCoords_vessel[crop][vessel] as double[], yCoords_vessel[crop][vessel] as double[], plane)
        // Create & new annotation & add it to the object hierarchy
        def vessel = new PathDetectionObject(roi_vessel, classes[(int) classes_index[crop][vessel]-1])
        imageData.getHierarchy().addPathObject(vessel, false)

        //def detection_vessel = PathObjects.createDetectionObject(roi_vessel)
        //addObject(detection_vessel, false)
        //detection_vessel.setPathClass(classes[(int) classes_index[crop][vessel]-1])
    }
}

resolveHierarchy()

//pngfile = Dialogs.promptForFile("Choose CCL21 png mask", null, "json",null)
pngfile = new File(dir + "mask_CCL21.png")
if (pngfile.exists()) {
    loadMask(imageData.getHierarchy(), pngfile,"CCL21");
}
//pngfile = Dialogs.promptForFile("Choose FAT png mask", null, "json",null)
pngfile =  new File(dir + "mask_fat.png")
if (pngfile.exists()) {
    loadMask(imageData.getHierarchy(), pngfile,"FAT");
}
//pngfile = Dialogs.promptForFile("Choose METASTASIS png mask", null, "json",null)
pngfile =  new File(dir + "mask_metastasis.png")
if (pngfile.exists()) {
    loadMask(imageData.getHierarchy(), pngfile,"METASTASIS");
}