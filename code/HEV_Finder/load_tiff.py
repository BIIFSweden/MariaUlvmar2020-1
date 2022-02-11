import qptiff, tifffile
import glob, os, json

import ROI

import numpy as np
import imageio
from skimage import morphology

inputDir = r"C:\Users\chrav452\Documents\Projects\MariaUlvmar2020-01\2020_10_10_original_database"
outputDir = r"C:\Users\chrav452\Documents\Projects\MariaUlvmar2020-01\2020_10_10_original_database\2020_10_21_dataCOCO\\"

os.makedirs(outputDir, exist_ok=True)
if os.path.isfile (outputDir + "\\" + "COCOannotations.json"):
    with open(outputDir + "\\" + "COCOannotations.json",'r') as JSONFile:
        COCOannotations = json.load(JSONFile)
else:
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
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "object","id": 1,"name": "not dilated vessel"},
            {"supercategory": "object","id": 2,"name": "Intermediately dilated vessel"},
            {"supercategory": "object","id": 3,"name": "highly dilated vessel"},
            {"supercategory": "tissue","id": 4,"name": "CCL21+"}
        ]
    }

#COCOannotations["images"] = [
#    i for i in COCOannotations["images"]
#    if not ("A3308-04" in i["file_name"] or "A11077" in i["file_name"] or "A5652" in i["file_name"] or "A7456" in i["file_name"] or "A14246" in i["file_name"])
#]

#with open(outputDir + "\\" + "COCOannotations.json",'w') as JSONFile:
#    json.dump(COCOannotations, JSONFile)
#for image in glob.glob(outputDir + "/**/*.png", recursive=True):
#    if "rgb_labels" in image or "annotations.png" in image:
#        continue
#    direct = os.path.dirname(image)
#    if not os.path.isfile(image.replace(".png","_rgb_labels.png")):
#        #print ("removing", image)
#        os.remove(image)
#for image in glob.glob(outputDir + "/**/*_rgb_labels.png", recursive=True):
#    os.remove(image)

# We parse all qptiff in subfolders of the inputDir
imageID, annotID = 0, 0
for qptiffFile in sorted(glob.glob(inputDir + r"\**\*.qptiff", recursive=True)):
    #if "with" in qptiffFile:
    #    continue
    folder = os.path.dirname(qptiffFile)
    relativeFolder = folder.replace(inputDir, "").replace("\\","/")
    if relativeFolder[0] == "/":
        relativeFolder = relativeFolder[1:]

    imagesInDB = [im for im in COCOannotations["images"] if relativeFolder == os.path.dirname(im["file_name"])]
    if len(imagesInDB) > 0:
        print ("This folder has already been processed")
        continue
    
    if os.path.isfile(outputDir + "\\" + relativeFolder + "\\annotations.json"):
        with open(outputDir + "\\" + relativeFolder + "\\annotations.json",'r') as JSONFile:
            annotationDicts = json.load(JSONFile)
    else:
        # We extract tiff images from the qptiff, using the annotationsFile
        try:
            annotationsFile = glob.glob(folder + r"\*_annotations.xml")[0]
        
            tiffile = qptiff.QPtiff(qptiffFile, outputDir + "\\" + relativeFolder)
            tiffile.load_annotations()
            whitelist = [
                int(i.split("_")[-1].replace(".zip",""))
                for i in glob.glob(folder + r"\*_*.zip")
            ]
            annotationDicts = tiffile.extractAnnotationsFromQptiff(whitelist = whitelist)
        except:
            print ("There was an error while processing qptiff file in " + folder)
            import traceback
            traceback.print_exc()
            continue

    with open(outputDir + "\\" + relativeFolder + "\\annotations.json",'w') as JSONFile:
        json.dump(annotationDicts, JSONFile)
        
    # For each annotation, we extract masks using ImageJ ROIs
    print ("reading ROIs")
    for annotationDict in annotationDicts:
        
        try:
            # We find the ROI file
            ROIFile = glob.glob(folder + r"\*_{key}.zip".format(key = annotationDict["key"]))[0]
        except:
            # No ROI for this image
            #if os.path.isfile(outputDir + "\\" + relativeFolder+"\\"+annotationDict["file_name"]):
            #    os.remove(outputDir + "\\" + relativeFolder+"\\"+annotationDict["file_name"])
            continue
        
        originalRGBImage = imageio.imread(outputDir + "\\" + relativeFolder+"\\"+annotationDict["file_name"])
        #originalTiffImage = tifffile.imread(outputDir + "\\" + relativeFolder+"\\"+annotationDict["path"])
        # We extract masks
        try:
            maskVessels, maskBackground, RGBImage, newCOCOAnnotations = ROI.roi_areas(
                ROIFile, 
                shape=[annotationDict["height"],annotationDict["width"]], 
                mpp=annotationDict["mpp"],
                originalRGB=originalRGBImage)
        except:
            continue    
        imageID = annotationDict["id"]
        COCOannotations["images"].append(
            {
                "file_name": relativeFolder+"/"+annotationDict["file_name"],
                "height": originalRGBImage.shape[0],
                "width": originalRGBImage.shape[1],
                "id": imageID,
                "global_x":annotationDict["global_x"],
                "global_y":annotationDict["global_y"],
            }
        )
        for COCOannotation in newCOCOAnnotations:
            annotID += 1
            COCOannotation["image_id"] = imageID
            COCOannotation["id"] = annotID
        COCOannotations["annotations"] += newCOCOAnnotations
        #continue
        # We save both masks as tiff images
        #filenameMask = '{key}_mask.tiff'.format(
        #    key = annotationDict["key"]
        #)
        #filenameMaskSeg = '{key}_mask_seg.png'.format(
        #    key = annotationDict["key"]
        #)
        #filenameMaskBackground = '{key}_mask_background.tiff'.format(
        #    key = annotationDict["key"]
        #)
        filenameRGBLabels = '{key}_rgb_labels.png'.format(
            key = annotationDict["key"]
        )
        #annotationDict["path_maskVessel"] = filenameMask
        #annotationDict["path_maskVesselSeg"] = filenameMaskSeg
        #annotationDict["path_maskBackground"] = filenameMaskBackground
        #annotationDict["path_rgbLabels"] = filenameRGBLabels
        
        #vesselsMarker = originalTiffImage[1:2,:,:].max(axis=0)
        #vesselsMarker[vesselsMarker<20] = 0
        #vesselsMarker[vesselsMarker>=20] = 255
        #vesselsMarker = morphology.binary_closing(vesselsMarker, np.ones((10, 10)))
        #vesselsMarker = morphology.binary_opening(vesselsMarker, np.ones((5, 5)))
        
        #vesselsMarker = vesselsMarker.astype(np.bool)
        #vesselsMarker = morphology.remove_small_holes(vesselsMarker, 2000)
        #vesselsMarker = morphology.remove_small_objects(vesselsMarker, 200)

        #tifffile.imsave(outputDir + "\\" + relativeFolder + "\\" + filenameMask, maskVessels)
        #maskVessels[vesselsMarker<=0] = 0
        
        #imageio.imsave (outputDir + "\\" + relativeFolder + "\\" + filenameMaskSeg, maskVessels.astype(np.uint8))
        #tifffile.imsave(outputDir + "\\" + relativeFolder + "\\" + filenameMaskBackground, maskBackground)
        imageio.imwrite(outputDir + "\\" + relativeFolder + "\\" + filenameRGBLabels, RGBImage)   
        
    with open(outputDir + "\\" + relativeFolder + "\\annotations.json",'w') as JSONFile:
        json.dump(annotationDicts, JSONFile)
    with open(outputDir + "\\" + "COCOannotations.json",'w') as JSONFile:
        json.dump(COCOannotations, JSONFile)

#COCOannotations["images"] = [
#    i for i in COCOannotations["images"]
#    if not ("A3308-04" in i["file_name"] or "A11077" in i["file_name"] or "A5652" in i["file_name"] or "A7456" in i["file_name"] or "A14246" in i["file_name"])
#]
for image in glob.glob(outputDir + "/**/*.png", recursive=True):
    if "rgb_labels" in image or "annotations.png" in image:
        continue
    direct = os.path.dirname(image)
    if not os.path.isfile(image.replace(".png","_rgb_labels.png")):
        #print ("removing", image)
        os.remove(image)
#for image in glob.glob(outputDir + "/**/*_rgb_labels.png", recursive=True):
#    os.remove(image)

with open(outputDir + "\\" + "COCOannotations.json",'w') as JSONFile:
    json.dump(COCOannotations, JSONFile)