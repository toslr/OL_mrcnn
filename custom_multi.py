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
from memory_profiler import profile

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


class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "multicell"


    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 4 # Background + opc, arborized, partial, ring

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    DETECTION_MIN_CONFIDENCE = 0.9
    
    LEARNING_RATE = 0.001

    #gpu = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpu[0], True)
    DEVICE = "/cpu:0"


    #IMAGE_CHANNEL_COUNT = 1
    #MEAN_PIXEL = np.array([127.0])



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
        self.add_class("multicell", 1, "ring")
        self.add_class("multicell", 2, "partial")
        self.add_class("multicell", 3, "arborized")
        self.add_class("multicell", 4, "opc")
     
        # Train or validation dataset?
        assert subset in ["train", "valid","test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations_dir = os.path.join(dataset_dir, "jsons")
        name_dict = {"ring":1, "partial":2, "arborized":3, "opc":4}

        for file in os.listdir(annotations_dir):
            if file.endswith(".json"):
                image_id = os.path.basename(file)[:-10]
                annotations = json.load(open(os.path.join(annotations_dir, file))) 
                #{'shapes':[{'points':[[],[]],'label':'Hard_hat'},{'points':[[],[]],'label':'Safety_vest'}],'imagePath':'0001.jpg','imageHeight':480,'imageWidth':640}
                point_sets = [shape['points'] for shape in annotations['shapes']]
                objects = [shape['label'] for shape in annotations['shapes']]

                num_ids = [name_dict[a] for a in objects]

                self.add_image(
                    source = "multicell",
                    image_id = image_id,
                    path = os.path.join(dataset_dir, "imgs", os.path.basename(file)[:-10]) +'.tif',
                    width = annotations['imageWidth'],
                    height = annotations['imageHeight'],
                    polygons = point_sets,
                    num_ids = num_ids
                ) # if using polygon filler
                '''self.add_image(
                    source = "cell",
                    image_id = image_id,
                    path = os.path.join(dataset_dir, "imgs", os.path.basename(file)[:-10]) +'.tif',
                    width = annotations['imageWidth'],
                    height = annotations['imageHeight'],
                    points = point_sets
                ) # if using full set of points'''

    '''def load_image(self, image_id):
        """Load the specified image and return a [H,W,1] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'], as_gray=True)*255
        image = np.expand_dims(image.astype(np.uint8), axis=2)
        return image'''

    def load_image(self, image_id):
        """Load the specified image as a fake RGB and return a [H,W,3] Numpy array.
        """
        # Load the image as grayscale
        gray_image = skimage.io.imread(self.image_info[image_id]['path'], as_gray=True)
        gray_image = (gray_image * 255).astype(np.uint8)

        # Stack the grayscale image into 3 channels
        rgb_image = np.stack((gray_image,)*3, axis=-1)
        return rgb_image

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        num_ids = image_info["num_ids"]
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
        num_ids = np.array(num_ids, dtype=np.int32)
        #mask = apply_voids(mask, binary_mask)
        return mask.astype(bool), num_ids

    '''def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cell":
            return super(self.__class__, self).load_mask(image_id)

        
        # Convert polygons to a bitmap mask of shape [height, width, instance_count]        
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(image_info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            all_points_x = [point[0] for point in p]
            all_points_y = [point[1] for point in p]
            rr, cc = skimage.draw.polygon(all_points_y,all_points_x)
            mask[rr, cc, i] = 1
        """
        # Convert points to a bitmap mask of shape [height, width, instance_count]
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["points"])])
        for i, set in enumerate(image_info["points"]):
            all_points_x = [point[0] for point in set]
            all_points_y = [point[1] for point in set]
            mask[all_points_y, all_points_x, i] = 1"""


        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)'''

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def nms_suppression_multi(results,threshold):
    """Apply non-maximum suppression to avoid multiple detections of the same object"""
    r = results[0]
    remove_index = np.array([0]*len(r['rois'])).astype(bool)
    for i in range(0,len(r['rois'])):
        for j in range(i+1,len(r['rois'])):
            if remove_index[i] == 0 and remove_index[j] == 0:
                iou = np.logical_and(r['masks'][:,:,i].astype(bool),r['masks'][:,:,j].astype(bool)).sum() / np.logical_or(r['masks'][:,:,i].astype(bool),r['masks'][:,:,j].astype(bool)).sum()
                #print(iou)
                if iou > threshold:
                    if r['scores'][i] > r['scores'][j]:
                        remove_index[j]=1
                    else:
                        remove_index[i]=1    
    new_results = [{'rois':r['rois'][~remove_index],
                    'masks':r['masks'][:,:,~remove_index],
                    'class_ids':r['class_ids'][~remove_index],
                    'scores':r['scores'][~remove_index]}]
    return new_results



@profile
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom("/Users/tom/Desktop/Stanford/RA/OligodendroSight/OL_mrcnn/data", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("/Users/tom/Desktop/Stanford/RA/OligodendroSight/OL_mrcnn/data", "valid")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
                
    '''model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='all',
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
                ))'''
	
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads_and_conv1',
                augmentation = imgaug.augmenters.Sequential([ 
                imgaug.augmenters.Fliplr(1), 
                imgaug.augmenters.Flipud(1), 
                imgaug.augmenters.Affine(rotate=(-45, 45)), 
                imgaug.augmenters.Affine(rotate=(-90, 90)), 
                imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                imgaug.augmenters.Crop(px=(0, 10)),
                imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                ]
                ))
    
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=10,
            layers='all',
            augmentation = imgaug.augmenters.Sequential([ 
            imgaug.augmenters.Fliplr(1), 
            imgaug.augmenters.Flipud(1), 
            imgaug.augmenters.Affine(rotate=(-45, 45)), 
            imgaug.augmenters.Affine(rotate=(-90, 90)), 
            imgaug.augmenters.Affine(scale=(0.5, 1.5)),
            imgaug.augmenters.Crop(px=(0, 10)),
            imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ]
            ))

    
# Another way of using imgaug    
# augmentation = imgaug.Sometimes(5/6,aug.OneOf(
                                            # [
                                            # imgaug.augmenters.Fliplr(1), 
                                            # imgaug.augmenters.Flipud(1), 
                                            # imgaug.augmenters.Affine(rotate=(-45, 45)), 
                                            # imgaug.augmenters.Affine(rotate=(-90, 90)), 
                                            # imgaug.augmenters.Affine(scale=(0.5, 1.5))
                                             # ]
                                        # ) 
                                   # )


if __name__ == '__main__':
    config = CustomConfig()

    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    weights_path = COCO_WEIGHTS_PATH # Download weights file
    
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)

    #model.load_weights(weights_path, by_name=True, exclude=['mrcnn_bbox_fc', 'mrcnn_class_logits','mrcnn_mask','conv1'])
    tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True, skip_mismatch=True)
        
    train(model)			