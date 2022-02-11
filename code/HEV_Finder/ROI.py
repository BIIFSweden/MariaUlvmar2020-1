"""this example is extremely simplistic and not very efficient, but it's easy to read."""
from read_roi import read_roi_zip
import os
import math
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage.draw import polygon

def containsLine (roi, roi_line):
    for (x,y) in [(roi_line["x1"],roi_line["y1"]), (roi_line["x2"],roi_line["y2"])]:
        point = Point(x,y)
        polygon = Polygon(zip(roi["x"],roi["y"]))
        if not polygon.contains(point):
            return False
    return True
def contains (roi1, roi2):
    for (x,y) in zip(roi2["x"][::20],roi2["y"][::20]):
        point = Point(x,y)
        polygon = Polygon(zip(roi1["x"],roi1["y"]))
        if not polygon.contains(point):
            return False
    return True

def roi_areas(roiFile, shape, mpp, originalRGB):
    """
    Given an ROI file (created with ImageJ's ROI Manager), analyzes each
    ROI and returns a mask and a label
    """
    mask_vessels = np.zeros(shape, dtype=np.int8)
    mask_background = np.zeros(shape, dtype=np.int8)
    original_shape = (911,1920)

    assert os.path.exists(roiFile)
    rois = read_roi_zip(roiFile)
    
    COCOannotations = []
    for roi_name, roi  in rois.items():
        #print (roi_name)
        # populate keys for rectangles
        if roi['type'] == 'freehand':
            lines = []
            label = None
            for roi_line in rois.values():
                if roi_line['type'] == 'line':
                    if containsLine(roi, roi_line):
                        lines.append(roi_line)
                elif roi_line != roi:
                    if contains(roi, roi_line):
                        label = 4
                        break
            else:
                if len(lines) == 0:
                    label = 1
                elif len(lines) <= 3:
                    lengths = [
                        math.sqrt(((line["x2"] - line["x1"]) ** 2) + ((line["y2"] - line["y1"]) ** 2))
                        for line in lines
                    ]
                    avgDistance = np.mean(lengths) * mpp
                    if avgDistance <= 10:
                        label = 2
                    else:
                        label = 3
                else:
                    print ("I got a ROI with more than 3 lines!", roiFile)
                    label = 4
            y_coor = np.array(roi["y"])*shape[0]/original_shape[0]
            y_coor = y_coor.tolist()
            #y_coor = y_coor[::10]
            x_coor = np.array(roi["x"])*shape[1]/original_shape[1]
            x_coor = x_coor.tolist()
            #x_coor = x_coor[::10]
            rr, cc = polygon(y_coor, x_coor, shape)
            # We remove empty ROIs:
            no_color = np.where(np.sum(originalRGB[rr, cc,:],axis=1)<70)[0]
            all_colors = np.where(np.sum(originalRGB[rr, cc,:],axis=1)<=255*3)[0]
            
            #print (len(no_color), len(all_colors), len(no_color) / len(all_colors), len(no_color) / len(all_colors) >= 0.95, roiFile)
            if len(roi_name) == 9: #label == 1 and len(no_color) / len(all_colors) >= 0.98:
                #print ("    TRUE")
                label = 4

            if label<= 3:
                mask_vessels[rr, cc] = label
            else:
                mask_background[rr, cc] = 1
            
            colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
            for x1, y1, x2, y2 in zip(x_coor, y_coor, x_coor[1:]+[x_coor[0]], y_coor[1:]+[y_coor[0]]):
                nb_steps = max(1, abs(y2-y1), abs(x2-x1))
                for step in range(int(nb_steps)+1):
                    x,y=int(x1+(step / nb_steps)*(x2-x1)), int(y1+(step / nb_steps)*(y2-y1))
                    try:
                        originalRGB[y, x] = colors[label-1]
                    except:
                        pass
            
            COCOannotations.append({
                "segmentation": [[pixel for t in zip(x_coor, y_coor) for pixel in t] ],
                "area": len(rr),
                "iscrowd": 0,
                "bbox": [min(x_coor),min(y_coor),max(x_coor)-min(x_coor),max(y_coor)-min(y_coor)],
                "category_id": label
            })
    
    return mask_vessels, mask_background, originalRGB, COCOannotations