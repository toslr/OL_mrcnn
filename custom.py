import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import imgaug
import tensorflow as tf
#from memory_profiler import profile

# Root directory of the project
ROOT_DIR = "/Users/tom/Desktop/Stanford/RA/OligodendroSight/OL_mrcnn"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Config
############################################################

GRAYSCALE = True
DATA_PATH = "/Users/tom/Desktop/Stanford/RA/OligodendroSight/OL_mrcnn/data_crop"

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    
    NAME = "cell" # Give the configuration a recognizable name

    GPU_COUNT = 1 # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    IMAGES_PER_GPU = 1 # 12GB GPU = 2 small images. Adjust down if you use a smaller GPU.
    
    NUM_CLASSES = 1 + 1   # Number of classes (including background=1)

    EPOCHS = 1 # Number of epochs to train
    STEPS_PER_EPOCH = 1 # Number of training steps per epoch
    LEARNING_RATE = 0.001 # Learning rate
    LAYERS = "all" # layers='heads' or 'all'

    DETECTION_MIN_CONFIDENCE = 0.9
    
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    MAX_GT_INSTANCES = 100 # Maximum number of ground truth instances to use in one image
    DETECTION_MAX_INSTANCES = 35 # Maximum number of instances in one image

    


    if DEVICE == "/gpu:0":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)



############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
      
        # Add classes. We have only one class to add.
        self.add_class("cell", 1, "ring")
     
        # Train / validation / test dataset?
        assert subset in ["train", "valid", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations_dir = os.path.join(dataset_dir, "jsons")

        for file in os.listdir(annotations_dir):
            if file.endswith(".json"):
                image_id = os.path.basename(file)[:-10]
                annotations = json.load(open(os.path.join(annotations_dir, file))) 
                #{'shapes':[{'points':[[],[]],'label':'my_label'},{'points':[[],[]],'label':'my_label'}],'imagePath':'0001.jpg','imageHeight':480,'imageWidth':640}
                point_sets = [shape['points'] for shape in annotations['shapes']]
                self.add_image(
                    source = "cell",
                    image_id = image_id,
                    path = os.path.join(dataset_dir, "imgs", os.path.basename(file)[:-10]) +'.tif',
                    width = annotations['imageWidth'],
                    height = annotations['imageHeight'],
                    polygons = point_sets
                ) # if using polygon filler

    def load_image(self, image_id):
        """Load the specified image as a fake RGB and return a [H,W,3] Numpy array.
        """
        if GRAYSCALE:
            # Load the image as grayscale
            gray_image = skimage.io.imread(self.image_info[image_id]['path'], as_gray=True)
            gray_image = (gray_image * 255).astype(np.uint8)

            # Stack the grayscale image into 3 channels
            rgb_image = np.stack((gray_image,)*3, axis=-1)
            return rgb_image
        else:
            return super().load_image(image_id)


    def load_mask(self, image_id):
        """ Load instance masks for the given image and returns a [H,W,instance_count] array of binary masks. 
        """
        image_info = self.image_info[image_id]
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])],
                        dtype=np.uint8)
        binary_mask_path = os.path.join(os.path.dirname(image_info["path"])[:-5], "masks", image_info["id"] + ".tif")
        binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
        def apply_voids(masks, base_image):
            void_mask = base_image > 0  # Assuming nnz values indicate voids
            for i in range(masks.shape[2]):
                masks[:, :, i][void_mask] = 0  # Apply voids to each object mask
            return masks
        for i,p in enumerate(image_info["polygons"]):
            all_points_x = [point[0] for point in p]
            all_points_y = [point[1] for point in p]
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr,cc,i] = 1
        mask = apply_voids(mask, binary_mask)
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image.
        """
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

#@profile
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(DATA_PATH, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(DATA_PATH, "valid")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
                
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS,
                layers=config.LAYERS, #) #layers='all', 
                augmentation = imgaug.augmenters.Sequential([ 
                imgaug.augmenters.Fliplr(1), 
                imgaug.augmenters.Flipud(1), 
                imgaug.augmenters.Affine(rotate=(-45, 45)), 
                imgaug.augmenters.Affine(rotate=(-90, 90)), 
                imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                imgaug.augmenters.Crop(px=(0, 10)),
                imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
                imgaug.augmenters.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                imgaug.augmenters.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                #imgaug.augmenters.Invert(0.05, per_channel=True), # invert color channels
                imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                
                ]
                
                ))
	

if __name__ == '__main__':
    config = CustomConfig()

    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    weights_path = COCO_WEIGHTS_PATH # Download weights file
    
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)

    #model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True, skip_mismatch=True)
        
    train(model)			