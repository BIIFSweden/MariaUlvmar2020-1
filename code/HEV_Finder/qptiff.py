import json, os
from xml.etree import ElementTree
import xmltodict
import tifffile
import imageio
import glob
from random import randrange,shuffle

from PIL import Image, ImageDraw
#from matplotlib.pyplot import imshow
import numpy as np
from skimage.filters import gaussian, threshold_otsu, threshold_local, apply_hysteresis_threshold
from skimage.measure import label, regionprops, find_contours, perimeter
from skimage.morphology import remove_small_holes, remove_small_objects, binary_opening, binary_closing, binary_erosion, binary_dilation, disk, opening, closing, dilation, erosion
import scipy

IN_MM = 0
IN_PIX = 1
class QPtiff():
    def __init__(self, qpPath, outputDir):
        self.qpPath = qpPath
        self.bounds = []
        self.outputDir = outputDir
        self.cocoFilename = self.outputDir + "/" + "COCOannotations.json"
        
    def load_annotations(self):
        xmlAnnotPath =  glob.glob(os.path.dirname(self.qpPath) + "/*.xml")
        if len(xmlAnnotPath) != 1:
            print ("Impossible to find xml annotation file, skipping {0}".format(self.qpPath))
            return
        path = xmlAnnotPath[0]
        print ("Loading annotation file {0}".format(path))
        toDelete = ["cad5d140-f7a2-45b9-a0f0-dd7a810915a2","88ada369-1e43-470a-a82b-25b9d8f3466e"] #A3308-04 2
        
        toDuplicate = ["775090cf-74c8-4038-845a-ed44ce4a77f5","c6ac92d5-eaf3-4f7e-ae22-0ad6fd5ea904", "26a3fe80-1784-4ef8-946f-4ffb8858b6be","0ad6c085-63b3-451a-8f86-c0533b7f8e6d", "0de9c00f-d3bf-474c-80f2-a3110dc8270b"]
        def load_json(path):  
            xml = ElementTree.tostring(ElementTree.parse(path).getroot())
            return xmltodict.parse(xml, attr_prefix="@", cdata_key="#text", dict_constructor=dict)

        annotations = load_json(path)
        annotations = annotations["AnnotationList"]["Annotations"]["Annotations-i"]

        for annotation in annotations:
            if "Deleted" in [h["Type"] for h in annotation["History"]["History-i"]]:
                continue
            if (float(annotation["Bounds"]["Size"]["Width"]) / float(annotation["Bounds"]["Size"]["Height"]) <= 1.6):
                continue
            if (annotation["ID"] in toDelete):
                continue
            self.bounds.append((
                    float(annotation["Resolution"]),
                    float(annotation["Bounds"]["Origin"]["X"]),
                    float(annotation["Bounds"]["Origin"]["Y"]),
                    float(annotation["Bounds"]["Size"]["Width"]),
                    float(annotation["Bounds"]["Size"]["Height"]),
                    IN_MM
                )
            )
            if (annotation["ID"] in toDuplicate):
                self.bounds.append(self.bounds[-1])
        print ("{0} annotations found in {1}".format(len(self.bounds), path))

    def getBoundsFromBlue(self, imageBlue, globalShape):
        def isRectangleOverlap(R1, R2):
            if (R1[0]>=R2[0]+R2[2]) or (R1[0]+R1[2]<=R2[0]) or (R1[1]+R1[3]<=R2[1]) or (R1[1]>=R2[1]+R2[3]):
                return False
            else:
                return True
        imageBlue = gaussian(imageBlue, sigma=3)
        #imageio.imsave("log1.png", imageBlue)
        thresh = threshold_otsu(imageBlue)
        #imageBlue[imageBlue < thresh] = 0
        #imageBlue[imageBlue >= thresh] = 255
        
        #imageio.imsave("log.png", imageBlue)
        bigw,bigh = 1920,911
        w,h = int(bigw*imageBlue.shape[1]/globalShape[1]),int(bigh*imageBlue.shape[0]/globalShape[0])
        bboxes = [
            (randrange(imageBlue.shape[1]-w),randrange(imageBlue.shape[0]-w),w,h) for i in range(1000)
        ]
        bboxes = [
            (x,y,w,h) for (x,y,w,h) in bboxes if len(np.where(imageBlue[y:y+h,x:x+w] > thresh)[0]) > 0.8*w*h
        ]
        intensities = [
            np.median(imageBlue[y:y+h,x:x+w]) for (x,y,w,h) in bboxes
        ]
        max_keep = 0
        sortedIndices = list(range(len(bboxes)))#np.argsort(intensities)
        for tries in range(100):
            keep = [1]*len(bboxes)
            shuffle(sortedIndices)
            for i in range(len(sortedIndices)):
                R1 = bboxes[sortedIndices[i]]
                for j in range(i+1,len(sortedIndices)):
                    R2 = bboxes[sortedIndices[j]]
                    if isRectangleOverlap(R1,R2):
                        keep[sortedIndices[i]] = 0
            if len([k for k in keep if k]) > max_keep:
                max_keep = len([k for k in keep if k])
                keep_ = keep
        bboxes = [
            b for b,k in zip(bboxes, keep_) if k
        ]
        for (x,y,w,h) in bboxes:
            self.bounds.append((
                    0.5,
                    x*globalShape[1]/imageBlue.shape[1],
                    y*globalShape[0]/imageBlue.shape[0],
                    float(bigw),
                    float(bigh),
                    IN_PIX
                )
            )

    def getBoundsFromCCL21(self, globalShape):
        
        imageBlue = self.image_CCL21
        
        bigw,bigh = 1920,911
        bboxes = []
        for x1 in range(0, globalShape[1], bigw):
            for y1 in range(0, globalShape[0], bigh):
                x2 = min(x1+bigw, globalShape[1])
                y2 = min(y1+bigh, globalShape[0])
                maskX1, maskX2 = int(x1*imageBlue.shape[1]/globalShape[1]), int(x2*imageBlue.shape[1]/globalShape[1])
                maskY1, maskY2 = int(y1*imageBlue.shape[0]/globalShape[0]), int(y2*imageBlue.shape[0]/globalShape[0])
                if np.sum(imageBlue[maskY1:maskY2,maskX1:maskX2]) > 0:
                    bboxes.append((x1,y1,x2-x1,y2-y1))
        for (x,y,w,h) in bboxes:
            self.bounds.append((
                    0.5,
                    x,
                    y,
                    float(bigw),
                    float(bigh),
                    IN_PIX
                )
            )

    def getBoundsFromAll(self, image_blue, globalShape):
        imageBlue = gaussian(image_blue, sigma=3)
        #imageio.imsave("log1.png", imageBlue)
        thresh = threshold_otsu(imageBlue)
        imageBlue[imageBlue < thresh] = 0
        bigw,bigh = 1920,911
        bboxes = []
        for x1 in range(0, globalShape[1], bigw):
            for y1 in range(0, globalShape[0], bigh):
                x2 = min(x1+bigw, globalShape[1])
                y2 = min(y1+bigh, globalShape[0])
                maskX1, maskX2 = int(x1*imageBlue.shape[1]/globalShape[1]), int(x2*imageBlue.shape[1]/globalShape[1])
                maskY1, maskY2 = int(y1*imageBlue.shape[0]/globalShape[0]), int(y2*imageBlue.shape[0]/globalShape[0])
                if np.sum(imageBlue[maskY1:maskY2,maskX1:maskX2]) > 0:
                    bboxes.append((x1,y1,x2-x1,y2-y1))
        for (x,y,w,h) in bboxes:
            self.bounds.append((
                    0.5,
                    x,
                    y,
                    float(bigw),
                    float(bigh),
                    IN_PIX
                )
            )
    def toRGB(self, imgArray):
        imgArray = np.array([imgArray[2,:,:],imgArray[1,:,:],imgArray[4,:,:]])
        imgArray = np.swapaxes(imgArray,0,2)
        imgArray = np.swapaxes(imgArray,0,1)
        return imgArray
    
    def extract_masks(self, imgArray, maskRes, CCL21Mask, fatMask, metastasisMask):
        print ("Mask extraction start")
        RGB = self.toRGB(imgArray)
        RGB = RGB * 3
        
        ## EXTRACT METASTASIS
        if metastasisMask:
            print ("   * Extracting metastasis mask")
            metaChannel = imgArray[3,:,:].astype(np.int32) - (imgArray[2,:,:].astype(np.int32)/2.)
            metaChannel[metaChannel<0] = 0
            #imageio.imsave('{output}/{filename}_metastasis_1.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), metaChannel[::2,::2])
            thresh = 80 #threshold_otsu(metaChannel)
            metaChannel[metaChannel < thresh] = 0
            metaChannel[metaChannel >= thresh] = 255
            metaChannel = binary_closing(metaChannel,disk(20))
            remove_small_objects(metaChannel, 800, in_place=True)
            WidthMicronsPixels = 200 / maskRes
            
            metaChannel = binary_dilation(metaChannel.astype("bool"),disk(int(WidthMicronsPixels)))

            imageio.imsave('{output}/mask_metastasis.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), metaChannel.astype(np.uint8) * 255)
            image_metastasis = RGB.copy()
            image_metastasis[metaChannel == 0,0] += 100

            imageio.imsave('{output}/{filename}_mask_metastasis_rgb.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), image_metastasis[::2,::2])
            

        ## EXTRACT CCL21
        if CCL21Mask:
            print ("   * Extracting CCL21 mask")
            image_blue = imgArray[4,:,:].astype(np.int16) - 0.5*imgArray[1,:,:].astype(np.int16)
            image_blue[image_blue <= 0] = 0
            image_blue = image_blue.astype(np.uint8)
            image_blue = gaussian(image_blue, sigma=15)
            thresh = threshold_otsu(image_blue)
            image_blue[image_blue<thresh] = 0
            image_blue[image_blue>=thresh] = 255
            image_blue = image_blue.astype('bool')
            remove_small_holes(image_blue, 1000000, in_place=True)
            image_blue = image_blue.astype(np.uint8)*255
            
            image_CCL21 = imgArray[4,:,:].astype(np.int16)# - 0.5*imgArray[1,:,:].astype(np.int16)
            image_CCL21[image_CCL21 <= 0] = 0
            
            image_CCL21 = image_CCL21.astype(np.uint8)
            
            from scipy import ndimage
            image_CCL21 = image_CCL21[::2, ::2]
            outVariance = ndimage.generic_filter(image_CCL21.astype(np.int16), np.var, footprint=disk(5))
            
            outVariance[outVariance>255] = 255
            outVariance = outVariance.astype(np.uint8)
            
            outVariance = gaussian(outVariance, sigma=2)*255.
            
            #outVariance = apply_hysteresis_threshold(outVariance, 30, 80)
            outVariance = apply_hysteresis_threshold(outVariance, 50, 80)
            outVariance = binary_erosion(outVariance,disk(2))
            
            remove_small_objects(outVariance, 300, in_place=True)
            
            outVariance = scipy.ndimage.zoom(outVariance, 2, order=0).astype(np.uint8)*255

            self.image_CCL21 = outVariance
            imageio.imsave('{output}/mask_CCL21.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), outVariance)
            image_CCL21 = RGB.copy()
            image_CCL21[outVariance == 0,0] += 100

            imageio.imsave('{output}/{filename}_mask_CCL21_rgb.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), image_CCL21[::2,::2])
        
        ## EXTRACT FAT
        if fatMask:
            print ("   * Extracting fat mask")
            allChannels = 3*imgArray[0,:,:].astype(np.int16)+imgArray[1,:,:].astype(np.int16)+imgArray[2,:,:].astype(np.int16)+imgArray[3,:,:].astype(np.int16)+imgArray[4,:,:].astype(np.int16)
            allChannels[allChannels<30] = 0
            allChannels[allChannels>=30] = 1
            allChannels = 1-allChannels
            allChannels = allChannels*255
            allChannels = allChannels.astype(np.uint8)
            allChannels[image_blue==False]=0 
            #imageio.imsave('{output}/fat_mask.png'.format(output=self.outputDir), allChannels)
            allChannels = allChannels.astype('bool')
            remove_small_objects(allChannels, 60, in_place=True)
            #print ("fat_mask_nosmall",allChannels.max(),allChannels.min())
            #imageio.imsave('{output}/fat_mask_nosmall.png'.format(output=self.outputDir), allChannels.astype(np.uint8)*255)
            #imageio.imsave('{output}/{filename}_rgb_original.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), RGB)
            RGB[allChannels>0] = [0,255,0]
            
            label_img = label(allChannels, connectivity=allChannels.ndim)
            props = regionprops(label_img)
        
            def angle(points):
                """
                Returns the angles between vectors.

                Parameters:
                dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

                The return value is a 1D-array of values of shape (N-1,), with each value
                between 0 and pi.

                0 implies the vectors point in the same direction
                pi/2 implies the vectors are orthogonal
                pi implies the vectors point in opposite directions
                """
                points1 = points[:-1]
                points2 = points[1:]
                
                dir = points2 - points1
                
                dir2 = dir[1:]
                dir1 = dir[:-1]
                return np.arccos((dir1*dir2).sum(axis=1)/(
                    np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

            def get_curvature (prop):
                b = np.zeros((prop.shape[0]+2,prop.shape[1]+2))
                b[1:-1,1:-1] = prop
                contours = find_contours(b, 0.5)
                total = 0
                for contour in contours:
                    #contour = contour[::10]
                    np.append(contour, contour[0])
                    angles = angle(contour)
                    if (len(angles) > 0):
                        total += float(np.sum(angles))/len(angles)
                return total
                
            for prop in props:
                #print (prop.intensity_image.sum(), prop.area, prop.intensity_image.sum() / prop.area)
                #print (prop.perimeter**2 / (4 * 3.14 * prop.area))
                opened1 = binary_opening(prop.image,disk(1))
                opened = binary_opening(prop.image,disk(4))
                if perimeter(opened) != 0:
                    diff_perim = perimeter(opened1) / perimeter(opened)
                else:
                    diff_perim = 1.

                #curv = get_curvature (prop.filled_image)
                #print (diff_perim)
                if diff_perim > 1.2:#0.65:
                #p2a = (prop.perimeter**2 / (4 * 3.14 * prop.area))
                #if p2a > 2:
                    for coordinates in prop.coords:                
                        allChannels[coordinates[0], coordinates[1]] = 0
                #exit()

            #imageio.imsave('{output}/fat_mask_final.png'.format(output=self.outputDir), allChannels.astype(np.uint8)*255)
            allChannels_opened = allChannels#binary_opening(allChannels.astype("bool"),disk(4))
            WidthMicronsPixels = 200 / maskRes
            
            allChannels_opened = binary_dilation(allChannels_opened.astype("bool"),disk(int(WidthMicronsPixels)))

            imageio.imsave('{output}/mask_fat.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), allChannels_opened.astype(np.uint8)*255)
            image_fat = RGB.copy()
            image_fat[allChannels_opened != 0,0] = 255
            imageio.imsave('{output}/{filename}_mask_fat_rgb.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), image_fat[::2,::2])
        print ("Mask extraction complete")
        #exit()
        #RGB[allChannels_opened>0] = [255,0,0]
        #imageio.imsave('{output}/{filename}_fat.png'.format(output=self.outputDir, filename=os.path.basename(self.qpPath)), RGB)

    def extractAnnotationsFromQptiff(self, whitelist = None, extractMethod="random", CCL21Mask=False, fatMask=False, metastasisMask=False):
        os.makedirs(self.outputDir, exist_ok=True)
        
        print ("Loading qptiff file", self.qpPath)
        with tifffile.TiffFile(self.qpPath, is_ome=False) as tif:
            xRes = 10000*tif.pages[0].tags["XResolution"].value[1] / tif.pages[0].tags["XResolution"].value[0]
            yRes = 10000*tif.pages[0].tags["YResolution"].value[1] / tif.pages[0].tags["YResolution"].value[0]
            xPos = 10000*tif.pages[0].tags["XPosition"].value[0] / tif.pages[0].tags["XPosition"].value[1]
            yPos = 10000*tif.pages[0].tags["YPosition"].value[0] / tif.pages[1].tags["YPosition"].value[1]
            
            maskXRes = 10000*tif.series[0].levels[3][0].tags["XResolution"].value[1] / tif.series[0].levels[3][0].tags["XResolution"].value[0]
            maskYRes = 10000*tif.series[0].levels[3][0].tags["YResolution"].value[1] / tif.series[0].levels[3][0].tags["YResolution"].value[0]

            self.extract_masks(tif.series[0].levels[3].asarray(),(maskXRes+maskYRes)/2.,CCL21Mask, fatMask, metastasisMask)
            
            image = tif.series[0]
            image = image.asarray()

            thumbArray = self.toRGB(tif.series[0].levels[3].asarray())
            image_thumb = Image.fromarray(thumbArray)
            if len(self.bounds) == 0:
                image_blue = thumbArray[:,:,2].astype(np.int16) - 0.5*thumbArray[:,:,1].astype(np.int16)
                image_blue[image_blue <= 0] = 0
                image_blue = image_blue.astype(np.uint8)
                if extractMethod == "allCCL21":
                    self.getBoundsFromCCL21(image.shape[1:])
                elif extractMethod == "random":
                    self.getBoundsFromBlue(image_blue, image.shape[1:])
                elif extractMethod == "all":
                    self.getBoundsFromAll(image_blue, image.shape[1:])
        #image_thumb = Image.new("RGB", (int(image.shape[2]/10), int(image.shape[1]/10)), color="white")
        draw = ImageDraw.Draw(image_thumb)

        print ("Extracting annotation images from qptiff", self.qpPath)
        self.annotationsDict = []
        for index, (res, x, y, w, h, mode) in enumerate(self.bounds):
            print (" - extracting annotation {0} out of {1}".format(index+1, len(self.bounds)))
            if whitelist is not None:
                if index+1 not in whitelist:
                    continue
            path_rgb = '{index}.png'.format(index=index+1)
            #path_ann = '{index}.tiff'.format(index=index+1)
            if mode == IN_MM:
                x, y, w, h = int((x-xPos)/xRes), int((y-yPos)/yRes), int(w/xRes), int(h/yRes)
            else:
                x, y, w, h = int(x), int(y), int(w), int(h)
            self.annotationsDict.append({
                #"path":path_ann, 
                "file_name":path_rgb, 
                "height":h,
                "width":w,
                "mpp":res,
                "key": index+1,
                "global_x":x,
                "global_y":y,
                "id":abs(hash(self.qpPath + "_" + path_rgb))
            })
            scale = [image_thumb.size[1]/image.shape[1], image_thumb.size[0]/image.shape[2]]
            
            xs, ys, ws, hs = int(x*scale[1]), int(y*scale[0]), int(w*scale[1]), int(h*scale[0])
            draw.rectangle([xs, ys, xs+ws, ys+hs], outline='white')
            draw.text((xs+5, ys+5), path_rgb, align ="left")  
            
            #if os.path.isfile(self.outputDir+"/"+path_rgb):
            #    print ("File exist, I skip.")
            #    continue
            
            image_ann = image[:,y:y+h,x:x+w]
            image_rgb = np.array([image_ann[2,:,:],image_ann[1,:,:],image_ann[4,:,:]])
            image_rgb = np.swapaxes(image_rgb,0,2)
            image_rgb = np.swapaxes(image_rgb,0,1)
            
            #tifffile.imsave(self.outputDir+"/"+path_ann, image_ann)
            try:
                imageio.imsave(self.outputDir+"/"+path_rgb, image_rgb)
            except:
                print ("Impossible to save annotation {0}".format(index+1))
                del self.annotationsDict[-1]
            

        #imshow(np.asarray(im))
        imageio.imsave('{output}/annotations.png'.format(output=self.outputDir), np.asarray(image_thumb))
        return self.annotationsDict
        
    def saveCOCO (self):    
        COCOannotations = {
            "info": {
                "description": "COCO 2017 Dataset",
                "   ": "https://www.scilifelab.se/facilities/bioimage-informatics/",
                "version": "1.0",
                "year": 2020,
                "contributor": "SciLifeLab BioImage Informatics",
                "date_created": "2020/10/04"
            },
            "licenses": [],
            "images": self.annotationsDict,
            "annotations": [],
            "categories": [
                {"supercategory": "object","id": 1,"name": "not dilated vessel"},
                {"supercategory": "object","id": 2,"name": "Intermediately dilated vessel"},
                {"supercategory": "object","id": 3,"name": "highly dilated vessel"},
                {"supercategory": "tissue","id": 4,"name": "CCL21+"}
            ]
        }
        
        with open(self.cocoFilename,'w') as JSONFile:
            json.dump(COCOannotations, JSONFile)



if __name__ == "__main__":
    import sys
    print (sys.argv[1])
    for filename in glob.glob(sys.argv[1],recursive=True):
        print (filename)
        try:
            qptiffObject = QPtiff(filename, sys.argv[2])
        
            annot = qptiffObject.extractAnnotationsFromQptiff()
            print (annot)
        except:
            import traceback
            traceback.print_exc()
        #qptiffObject.saveCOCO()
        #cocoFilename = qptiffObject.cocoFilename
