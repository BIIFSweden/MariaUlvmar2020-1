from PIL import Image
import os
import json

def ImagesToCOCO (images, outputDir):
    os.makedirs(outputDir + "/DL_Results/", exist_ok=True)
    COCOannotations = {
        "info": {
            "description": "COCO 2020 Dataset",
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
    for image in images:
        im = Image.open(image)
        w, h = im.size
        COCOannotations["images"].append({
            #"path":path_ann, 
            "file_name":image.replace("\\","/").replace(outputDir,"../"), 
            "height":h,
            "width":w,
            "global_x":0,
            "global_y":0,
            "id":abs(hash(image))
        })
    cocoFilename = outputDir + "/DL_Results/" + "COCOannotations.json"
    
    with open(cocoFilename,'w') as JSONFile:
        json.dump(COCOannotations, JSONFile)
    return cocoFilename