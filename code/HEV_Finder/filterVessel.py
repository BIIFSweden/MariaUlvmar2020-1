import json
import numpy as np
import imageio
import copy
import os.path
from skimage.measure import find_contours
from shapely.geometry import Polygon

def get_contour (mask,indexAnnot):
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    #imageio.imsave(r"C:\Users\chrav452\Documents\Projects\MariaUlvmar2020-01\2020_10_10_original_database\IDC with metastasis\A343-16 B\DL_results" + "/" + "tmp_" + str(indexAnnot+1) + ".png", padded_mask)
    
    contours = find_contours(padded_mask, 0.5)
    if len(contours) == 0:
        return []

    verts = np.fliplr(contours[0]) - 1
    return verts

def removeBorderVessels(jsonDescriptionVessels):
    #jsonDescription = {"xCoords_crop":[],"yCoords_crop":[],"xCoords_vessel":[],"yCoords_vessel":[],"classes":[],"scores":[]}
    newJsonDescriptionVessels = copy.deepcopy(jsonDescriptionVessels)
    nbAnnotations = len(jsonDescriptionVessels["xCoords_crop"])
    for indexAnnot in range(nbAnnotations):
        newJsonDescriptionVessels["xCoords_vessel"][indexAnnot] = []
        newJsonDescriptionVessels["yCoords_vessel"][indexAnnot] = []
        newJsonDescriptionVessels["classes"][indexAnnot] = []
        newJsonDescriptionVessels["scores"][indexAnnot] = []
        [xmin,xmax,_,_] = jsonDescriptionVessels["xCoords_crop"][indexAnnot]
        [ymin,ymax,_,_] = jsonDescriptionVessels["yCoords_crop"][indexAnnot]
        nbVessels = len(jsonDescriptionVessels["classes"][indexAnnot])
        for vesselIndex in range(nbVessels):
            xvessel = jsonDescriptionVessels["xCoords_vessel"][indexAnnot][vesselIndex]
            yvessel = jsonDescriptionVessels["yCoords_vessel"][indexAnnot][vesselIndex]
            isBorder = False
            for margin in [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]:
                if xmin+margin in xvessel or xmax+margin in xvessel or ymin+margin in yvessel or ymax+margin in yvessel:
                    isBorder = True
            if isBorder:
                continue
            for key in ["xCoords_vessel","yCoords_vessel","classes","scores"]:
                newJsonDescriptionVessels[key][indexAnnot].append(jsonDescriptionVessels[key][indexAnnot][vesselIndex])
            
    return newJsonDescriptionVessels

def removeSmallVessels(jsonDescriptionVessels, sizeMin):
    #jsonDescription = {"xCoords_crop":[],"yCoords_crop":[],"xCoords_vessel":[],"yCoords_vessel":[],"classes":[],"scores":[]}
    newJsonDescriptionVessels = copy.deepcopy(jsonDescriptionVessels)
    nbAnnotations = len(jsonDescriptionVessels["xCoords_crop"])
    for indexAnnot in range(nbAnnotations):
        newJsonDescriptionVessels["xCoords_vessel"][indexAnnot] = []
        newJsonDescriptionVessels["yCoords_vessel"][indexAnnot] = []
        newJsonDescriptionVessels["classes"][indexAnnot] = []
        newJsonDescriptionVessels["scores"][indexAnnot] = []
        
        nbVessels = len(jsonDescriptionVessels["classes"][indexAnnot])
        for vesselIndex in range(nbVessels):
            xvessel = jsonDescriptionVessels["xCoords_vessel"][indexAnnot][vesselIndex]
            yvessel = jsonDescriptionVessels["yCoords_vessel"][indexAnnot][vesselIndex]
            p1 = Polygon(zip(xvessel,yvessel))
            area = p1.area
            if area<sizeMin:
                continue
            for key in ["xCoords_vessel","yCoords_vessel","classes","scores"]:
                newJsonDescriptionVessels[key][indexAnnot].append(jsonDescriptionVessels[key][indexAnnot][vesselIndex])
            
    return newJsonDescriptionVessels

def vesselOverlap (x1, y1, x2, y2):
    p1 = Polygon(zip(x1,y1))
    p2 = Polygon(zip(x2,y2))
    return p1.intersects(p2)

def removeOverlapVessels(jsonDescriptionVessels):
    newJsonDescriptionVessels = copy.deepcopy(jsonDescriptionVessels)
    nbAnnotations = len(jsonDescriptionVessels["xCoords_crop"])
    for indexAnnot in range(nbAnnotations):
        newJsonDescriptionVessels["xCoords_vessel"][indexAnnot] = []
        newJsonDescriptionVessels["yCoords_vessel"][indexAnnot] = []
        newJsonDescriptionVessels["classes"][indexAnnot] = []
        newJsonDescriptionVessels["scores"][indexAnnot] = []
        nbVessels = len(jsonDescriptionVessels["classes"][indexAnnot])
        removeIndices = []
        for vesselIndex1 in range(nbVessels):
            for vesselIndex2 in range(nbVessels):
                if vesselIndex1 == vesselIndex2 or vesselIndex1 in removeIndices or vesselIndex2 in removeIndices:
                    continue
                overlap = vesselOverlap(
                    jsonDescriptionVessels["xCoords_vessel"][indexAnnot][vesselIndex1],
                    jsonDescriptionVessels["yCoords_vessel"][indexAnnot][vesselIndex1],
                    jsonDescriptionVessels["xCoords_vessel"][indexAnnot][vesselIndex2],
                    jsonDescriptionVessels["yCoords_vessel"][indexAnnot][vesselIndex2]
                )
                if overlap:
                    if jsonDescriptionVessels["scores"][indexAnnot][vesselIndex1]>jsonDescriptionVessels["scores"][indexAnnot][vesselIndex2]:
                        removeIndices.append(vesselIndex2)
                    else:
                        removeIndices.append(vesselIndex1)
        for vesselIndex in range(nbVessels):
            if not vesselIndex in removeIndices:
                for key in ["xCoords_vessel","yCoords_vessel","classes","scores"]:
                    newJsonDescriptionVessels[key][indexAnnot].append(jsonDescriptionVessels[key][indexAnnot][vesselIndex])
    return newJsonDescriptionVessels

def filterVessels(inputFolder, sizeMin):
    with open(inputFolder + "/qupath_rois.json", "r") as  f:
        jsonDescriptionVessels = json.load(f)
    mask_CCL21, mask_met, mask_fat = False, False, False
    if os.path.isfile(inputFolder + "/" + "mask_CCL21.png"):
        mask_CCL21 = imageio.imread(inputFolder + "/" + "mask_CCL21.png")
    if os.path.isfile(inputFolder + "/" + "mask_metastasis.png"):
        mask_met   = imageio.imread(inputFolder + "/" + "mask_metastasis.png")
    if os.path.isfile(inputFolder + "/" + "mask_fat.png"):
        mask_fat   = imageio.imread(inputFolder + "/" + "mask_fat.png")
    jsonDescriptionVesselsOK = copy.deepcopy(jsonDescriptionVessels)
    nbAnnotations = len(jsonDescriptionVessels["xCoords_crop"])
    jsonDescriptionVessels = removeBorderVessels(jsonDescriptionVessels)
    jsonDescriptionVessels = removeOverlapVessels(jsonDescriptionVessels)
    jsonDescriptionVessels = removeSmallVessels(jsonDescriptionVessels, sizeMin)
    
    ccl21_ratios = []
    for indexAnnot in range(nbAnnotations):
        nbVessels = len(jsonDescriptionVessels["classes"][indexAnnot])
        for vesselIndex in range(nbVessels):
            try:
                bbox = np.array((
                    (min(jsonDescriptionVessels["xCoords_vessel"][indexAnnot][vesselIndex]) + max(jsonDescriptionVessels["xCoords_vessel"][indexAnnot][vesselIndex])) / 2.,
                    (min(jsonDescriptionVessels["yCoords_vessel"][indexAnnot][vesselIndex]) + max(jsonDescriptionVessels["yCoords_vessel"][indexAnnot][vesselIndex])) / 2.
                ))
                bboxMaskCoord = bbox / 8.
                bboxMaskCoord = bboxMaskCoord.astype(np.uint16)
                if type(mask_CCL21) != type(True):
                    ccl21  = mask_CCL21[bboxMaskCoord[1],bboxMaskCoord[0]] != 0
                    if not(ccl21):
                        jsonDescriptionVessels["classes"][indexAnnot][vesselIndex] = 4
                if type(mask_fat) != type(True):
                    no_fat = mask_fat[bboxMaskCoord[1],bboxMaskCoord[0]] == 0
                    if not(no_fat):
                        jsonDescriptionVessels["classes"][indexAnnot][vesselIndex] = 5
                if type(mask_met) != type(True):
                    no_met = mask_met[bboxMaskCoord[1],bboxMaskCoord[0]] == 0
                    if not(no_met):
                        jsonDescriptionVessels["classes"][indexAnnot][vesselIndex] = 6
                    #jsonDescriptionVesselsOK["classes"][indexAnnot].append(jsonDescriptionVessels["classes"][indexAnnot][vesselIndex])
                    #jsonDescriptionVesselsOK["xCoords_vessel"][indexAnnot].append(jsonDescriptionVessels["xCoords_vessel"][indexAnnot][vesselIndex])
                    #jsonDescriptionVesselsOK["yCoords_vessel"][indexAnnot].append(jsonDescriptionVessels["yCoords_vessel"][indexAnnot][vesselIndex])
            except:
                import traceback
                traceback.print_exc()
        x1 = int(jsonDescriptionVessels["yCoords_crop"][indexAnnot][0] / 8.)
        x2 = int(jsonDescriptionVessels["yCoords_crop"][indexAnnot][2] / 8.)
        y1 = int(jsonDescriptionVessels["xCoords_crop"][indexAnnot][0] / 8.)
        y2 = int(jsonDescriptionVessels["xCoords_crop"][indexAnnot][2] / 8.)
        if (type(mask_CCL21) != type(True)):
            ccl21_annot  = mask_CCL21[x1:x2+1, y1:y2+1]
            
            #imageio.imsave(inputFolder + "/" + "tmp_CLL21_" + str(indexAnnot+1) + ".png", ccl21_annot)
            #try:
            #contour = get_contour (ccl21_annot,indexAnnot) * 8.
            try:
                ccl21_ratio = len(np.nonzero(ccl21_annot)[0]) / (ccl21_annot.shape[0] * ccl21_annot.shape[1])
                ccl21_ratios.append(ccl21_ratio)
                print (ccl21_ratio)
                #no_fat_annot = mask_fat[x1:x2+1, y1:y2+1]
                #no_met_annot = mask_met[x1:x2+1, y1:y2+1]
            except:
                ccl21_ratios.append(0)
        else:
            ccl21_ratios.append(0)
    
    lines = [["annotation","non dilated","semi dilated","fully dilated","CCL21-","close to fat","close to metastasis","CCL21+ ratio"]]
    nbAnnotations = len(jsonDescriptionVessels["xCoords_crop"])
    for indexAnnot in range(nbAnnotations):
        lines.append([indexAnnot+1,0,0,0,0,0,0,ccl21_ratios[indexAnnot]])
        nbVessels = len(jsonDescriptionVessels["classes"][indexAnnot])
        for vesselIndex in range(nbVessels):
            lines[-1][jsonDescriptionVessels["classes"][indexAnnot][vesselIndex]] += 1
    import csv

    with open(inputFolder + "vesselCount.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lines)
    
    with open(inputFolder + "/qupath_rois.json", "w") as  f:
        json.dump(jsonDescriptionVessels, f)
   
if __name__ == "__main__":
    import sys
    #print (sys.argv[1])
    filterVessels(sys.argv[1], 600)