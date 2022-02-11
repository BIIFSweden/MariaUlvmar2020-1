"""
Mask R-CNN
Train on the vessels segmentation dataset

Licensed under the MIT License (see LICENSE for details)
Written by Christophe Avenel based on code from Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 vessel.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 vessel.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 vessel.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 vessel.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import imageio 

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/vessel/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = list(range(15))


############################################################
#  Configurations
############################################################

class VesselConfig(Config):
    """Configuration for training on the vessel segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "vessels"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + vessel

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between vessel and BG
    #DETECTION_MIN_CONFIDENCE = 0
    DETECTION_MIN_CONFIDENCE = 0.7

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 0.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([7.634, 8.477, 15.605])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class VesselInferenceConfig(VesselConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.2
    DETECTION_MIN_CONFIDENCE = 0


############################################################
#  Dataset
############################################################

class VesselDataset(utils.Dataset):

    def load_vessel(self, dataset_dir, subset):
        """Load a subset of the vessels dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """

        coco = COCO("{}/COCOannotations.json".format(dataset_dir))
        image_dir = dataset_dir
        # Load all classes or a subset?
        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("vessels", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            if (subset == "val" and i in VAL_IMAGE_IDS) or (subset != "val" and i not in VAL_IMAGE_IDS):
                if coco.imgs[i]['file_name'][0] == "/":
                    coco.imgs[i]['file_name'] = coco.imgs[i]['file_name'][1:]
                self.add_image(
                    "vessels", image_id=i,
                    path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annotations=coco.loadAnns(coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))
                
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "vessels." + str(annotation['category_id'])
            )
            if class_id and class_id != 4:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        return info["url"]

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = VesselDataset()
    dataset_train.load_vessel(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VesselDataset()
    dataset_val.load_vessel(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])
    
    #train_generator = modellib.DataGenerator(dataset_train, model.config, shuffle=True,
    #                                     augmentation=augmentation)
    #x,y = train_generator[0]
    #print (x)
    #import matplotlib.pyplot as plt
    #from matplotlib.colors import ListedColormap
    #cMap = ListedColormap(['white', 'red', 'green','blue'])
    #for i in range(0,2):
    #    image = x[i]
    #    print (x[i].shape)
    #    print (y[i].shape)
    #    #plt.figure()
    #    #plt.imshow(image, cmap=cMap)
    #    #image = y[i]
    #    #print (image.shape, image.dtype, image[:].max(), image[:].min(), image[:].mean(), np.unique(image[:]))
    #    #plt.figure()
    #    #plt.imshow(image.squeeze(), cmap=cMap, vmax=N_CLASSES - 0.5, vmin=-0.5)
    #    #plt.colorbar()

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    #model.train(dataset_train, dataset_val,
    #            learning_rate=config.LEARNING_RATE,
    #            epochs=20,
    #            augmentation=augmentation,
    #            layers='heads', 
    #            custom_callbacks=[])

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=400,
                augmentation=augmentation,
                layers='all', 
                custom_callbacks=[])


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = VesselDataset()
    dataset.load_vessel(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        
        #masks_gt, class_gt = dataset.load_mask(image_id)
        #roi_gt = utils.extract_bboxes(masks_gt)
        #scores_gt = [0 for r in roi_gt]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'], # roi_gt, masks_gt, class_gt, #
            dataset.class_names, r['scores'], # scores_gt
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
    
def detect_images(model, cocoDatabase, imagePath):
    from skimage.measure import find_contours
    with open(cocoDatabase + "COCOannotations.json") as cocoFile:
        images = json.load(cocoFile)
    images = [i for i in images["images"] if imagePath in i["file_name"]]
    print (images)
    imageFolder = os.path.dirname(cocoDatabase)
    
    jsonDescription = {"xCoords_crop":[],"yCoords_crop":[],"xCoords_vessel":[],"yCoords_vessel":[],"classes":[]}
    
    #xCoords_crop = [[0, 300, 300, 0], [100, 400, 400, 100]] //origin, width, width, origin
    #yCoords_crop = [[0, 0, 50, 50], [70, 70, 150, 150]] //origin, origin, height, height

    #xCoords_vessel = [[[5, 20, 100, 200], [10, 40, 200, 300]], [[100, 150, 100, 200, 300]]]
    #yCoords_vessel = [[[10, 40, 45, 20], [10, 40, 45, 20]], [[75, 80, 80, 90, 120]]]
    
    # Path to a specific weights file
    for i, cocoImage in enumerate(images):
        jsonDescription["xCoords_crop"].append([cocoImage["global_x"], cocoImage["global_x"]+cocoImage["width"], cocoImage["global_x"]+cocoImage["width"], cocoImage["global_x"]])
        jsonDescription["yCoords_crop"].append([cocoImage["global_y"], cocoImage["global_y"], cocoImage["global_y"]+cocoImage["height"], cocoImage["global_y"]+cocoImage["height"]])
        jsonDescription["xCoords_vessel"].append([])
        jsonDescription["yCoords_vessel"].append([])
        image = imageio.imread(imageFolder + "/" + cocoImage["file_name"].replace("\\","/"))
        print ("Run detection image " + str(i) + " on " + str(len(images)))
        # Run detection
        results = model.detect([image], verbose=0)
        r = results[0]
        jsonDescription["classes"].append(list(r["class_ids"]))
        for i in range(r["rois"].shape[0]):
            if r["scores"] is not None:
                if r["scores"][i] < 0.5:
                    continue
            if not np.any(r["rois"][i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            mask = r["masks"][:,:,i]
            
            padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            
            contours = find_contours(padded_mask, 0.5)
            verts = np.fliplr(contours[0]) - 1
            verts += [cocoImage["global_x"], cocoImage["global_y"]]
            jsonDescription["xCoords_vessel"][-1].append(list(verts[::20,0].astype(np.int32)))
            jsonDescription["yCoords_vessel"][-1].append(list(verts[::20,1].astype(np.int32)))
    with open(cocoDatabase + "qupath_rois.json",'w') as f:
        json.dump(jsonDescription, f, cls=npEncoder)
    #print (jsonDescription)
############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse
    #import tensorflow as tf

    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #cpus = tf.config.experimental.list_physical_devices('CPU')

    #print (cpus,gpus)
    #if gpus:
    #  # Restrict TensorFlow to only use the first GPU
    #  try:
    #    tf.config.experimental.set_visible_devices(cpus[:])
    #    #tf.config.experimental.set_visible_devices(gpus, 'GPU')
    #  except RuntimeError as e:
    #    # Visible devices must be set at program startup
    #    print(e)
        
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for vessels counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--images', required=False,
                        metavar="path/to/images/",
                        help='Path to images in the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"
    elif args.command == "predict":
        assert args.images, "Provide --images to run prediction on"


    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VesselConfig()
    else:
        config = VesselInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    model.keras_model.save(args.dataset + "/model_maskRCNN")
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "predict":
        detect_images(model, args.dataset, args.images)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
