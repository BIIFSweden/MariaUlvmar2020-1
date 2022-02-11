#!/usr/bin/env python 

import os
import struct
import numpy as np
import tempfile
import zipfile
import codecs
import shutil
from skimage.measure import find_contours

def int_to_hex(i): # for 16 bit integers
    lo = i // 256
    hi = i % 256
    return [lo,hi]

def float_to_hex(f): # for 32 bit floats
    return (hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:]).decode('hex')
    

'''
Definition of the ROI format
https://imagej.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html
https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
 ImageJ/NIH Image 64 byte ROI outline header
    2 byte numbers are big-endian signed shorts
    
    0-3     "Iout"
    4-5     version (>=217)
    6-7     roi type
    8-9     top
    10-11   left
    12-13   bottom
    14-15   right
    16-17   NCoordinates
    18-33   x1,y1,x2,y2 (straight line)
    34-35   stroke width (v1.43i or later)
    36-39   ShapeRoi size (type must be 1 if this value>0)
    40-43   stroke color (v1.43i or later)
    44-47   fill color (v1.43i or later)
    48-49   subtype (v1.43k or later)
    50-51   options (v1.43k or later)
    52-52   arrow style or aspect ratio (v1.43p or later)
    53-53   arrow head size (v1.43p or later)
    54-55   rounded rect arc size (v1.43p or later)
    56-59   position
    60-63   header2 offset
    64-     x-coordinates (short), followed by y-coordinates
'''

def save_imagej_roi (x_coordinates,y_coordinates,name,path,stroke_width=3,stroke_col='FFFF0000',fill_col='00000000'):
    
    # Creates ImageJ compatible ROI files from masks
    # Parameters:
    #    x_coordinates:        List of X coordinates
    #    y_coordinates:        List of Y coordinates
    #    name:                Name of the ROI
    #    path:                path to write the output file
    # Optional parameters:
    #    stroke_width [int]
    #    stroke_col [0xAARRGGBB]
    #    fill_col   [0xAARRGGBB]
    #
    # Return: path to output file

    type=[0x07,0x00]
    # Find bounding box of ROI
    top,left,bottom,right=min(y_coordinates), min(x_coordinates), max(y_coordinates),max(x_coordinates)
    
    HEADER_SIZE, HEADER2_SIZE = 64,64
    filelength=HEADER_SIZE + 2*len(x_coordinates) + 2*len(y_coordinates) + HEADER2_SIZE + len(name)*2

    data=bytearray(filelength)

    data[0:4]=[0x49,0x6F,0x75,0x74]                       #"Iout" 0-3
    data[4:6]=[0x00,0xE3]                                 #Version 4-5

    data[6:8]=type                                  # roi type   6-7     # Ovals/points
    data[8:10] = int_to_hex(top)                     # top      8-9    
    data[10:12]= int_to_hex(left)                    # left    10-11   
    data[12:14]= int_to_hex(bottom)                  # bottom  12-13    
    data[14:16]= int_to_hex(right)                   # right   14-15
    data[16:18]= int_to_hex(len(x_coordinates))       # n   16-17
    data[34:36]= int_to_hex(stroke_width)              # Stroke Width  34-35  
    data[40:44]= bytes.fromhex(stroke_col)            # Stroke Color 40-43 
    data[44:48]= bytes.fromhex(fill_col)
    data[50:52] = [0x20,0x00]                        # SCALE_STROKE_WIDTH 
    
    Header2Offset = HEADER_SIZE + 2*len(x_coordinates) + 2*len(y_coordinates)
    data[60:64]= [0x00,0x00] + int_to_hex(Header2Offset)                    # header2offset 60-63   
    for index, (x, y) in enumerate(zip(x_coordinates, y_coordinates)):
        x, y = x-left, y-top
        start_x = HEADER_SIZE + 2 * index
        start_y = start_x + 2*len(x_coordinates)
        data[start_x:start_x+2] = int_to_hex(x)
        data[start_y:start_y+2] = int_to_hex(y)
    
    # add name
    p=Header2Offset                                
    for c in name:
        data[p:p+2]=codecs.encode(bytes(c, 'utf-8'), "hex")
        p=p+2
    
    # write file
    file = open(path,'wb')
    file.write(data)
    file.close()
    return(path)


def save_ImageJ_ROIs(masks, classes, output, keep):
    zip_path = output
    zip_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zfd:
        for index in range(masks.shape[2]):
            if not keep[index]:
                continue
            roi_name = "roi_{0}".format(index+1)
            roi_path = os.path.join(zip_dir, roi_name) + ".roi"
            #create roi_path file
            mask = masks[:,:,index]

            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            
            contours = find_contours(padded_mask, 0.5)
            if len(contours) == 0:
                continue
            verts = np.fliplr(contours[0]) - 1
            verts = verts.astype(np.uint16)
            coordinates_x = verts[:,0] #[::20,0]
            coordinates_y = verts[:,1] #[::20,1]
            if (classes[index] == 1):
                stroke_col='FFFF0000'
            elif (classes[index] == 2):
                stroke_col='FF00FF00'
            elif (classes[index] == 3):
                stroke_col='FF0000FF'
            save_imagej_roi (
                x_coordinates = coordinates_x,
                y_coordinates = coordinates_y,
                name = roi_name,
                path = roi_path,
                stroke_width = 3,
                stroke_col = stroke_col,
                fill_col = '00000000'
            )
            zfd.write(roi_path, roi_name + ".roi")
    shutil.rmtree(zip_dir)
    

#masks = np.load('./mask_example.npy')
#print (masks.shape)
#get_ImageJ_ROIs(masks, [], './mask_example_rois.zip')