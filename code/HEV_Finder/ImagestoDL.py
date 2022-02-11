from ImageSlicer import ImageSlicer
import glob, os
import imageio

inputDir = r"C:\Users\chrav452\Documents\Projects\MariaUlvmar2020-01\data"
outputDir = r"C:\Users\chrav452\Documents\Projects\MariaUlvmar2020-01\dldata"

os.makedirs(outputDir+'/masks/all', exist_ok=True)
os.makedirs(outputDir+'/images/all', exist_ok=True)

for maskfile in sorted(glob.glob(inputDir + r"\**\*_mask_seg.png", recursive=True)):
    break
    folder = os.path.dirname(maskfile)
    qptiffName = os.path.basename(folder)
    originalRGB = maskfile.replace("_mask_seg","")
    index = os.path.basename(originalRGB).replace(".png","")


    print (qptiffName + "_" + index)
    slicer = ImageSlicer(source=maskfile, size=(256,256), strides=(128,128)) #Provide image path and slice size you desire
    transformed_image = slicer.transform()
    slicer.save_images(transformed_image, outputDir+'/masks/all', qptiffName + "_" + index) #Provide the directory where you want to save the sliced images

    originalRGB = maskfile.replace("_mask_seg","")
    slicer = ImageSlicer(source=originalRGB, size=(256,256), strides=(128,128)) #Provide image path and slice size you desire
    transformed_image = slicer.transform()
    slicer.save_images(transformed_image, outputDir+'/images/all', qptiffName + "_" + index) #Provide the directory where you want to save the sliced images

for maskfile in glob.glob(outputDir+'/masks/all/*.png'):
    
    image = imageio.imread(maskfile)
    print (image[:].max())
    if (image[:].max() == 0):
        RGBimage = maskfile.replace ("/masks","/images")
        os.remove(maskfile)
        os.remove(RGBimage)
