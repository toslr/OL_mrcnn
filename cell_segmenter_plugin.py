import tensorflow as tf

# Creates a session with device placement logs
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

import warnings
warnings.filterwarnings('ignore')
import os
import shutil
import sys
import json
import datetime
import numpy as np
import skimage
import skimage.draw
import cv2
import random
import math
import re
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import tifffile as tiff
import argparse
import logging
logging.getLogger('tensorflow').addFilter(lambda record: "map_fn_v2" not in record.getMessage())
logging.getLogger('tensorflow').addFilter(lambda record: "op: \"CropAndResize\"" not in record.getMessage())

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from postprocessing import apply_mask_to_imgs

def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) #os.getcwd()
    sys.path.append(ROOT_DIR)  # To find local version of the library
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='/cpu:0', help='Device to use')
    parser.add_argument('--g', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--ci', type=float, default=0.7, help='Minimum confidence level for detection')
    parser.add_argument('--nms', type=float, default=0.3, help='NMS threshold for detection')
    parser.add_argument('--weights', type=str, default='mask_rcnn_coco.h5', help='Subpath to weights file')
    parser.add_argument('--data', type=str, default='data/test/imgs', help='Directory containing test images')
    parser.add_argument('--gray', action='store_true', help='Whether to convert images to grayscale')
    parser.add_argument('--multiclass', action='store_true', help='Whether to use the multiclass model')
    args = parser.parse_args()

    IMG_DIR_CROP = os.path.join(args.data, 'crops')
    IMG_DIR_NORM = os.path.join(args.data, 'crops_norm')
    MASK_DIR = os.path.join(args.data, 'masks')
    os.makedirs(MASK_DIR, exist_ok=True)

    GRAYSCALE = args.gray

    if args.multiclass:
        from custom_multi import CustomConfig, CustomDataset
    else:
        from custom import CustomConfig, CustomDataset

    class InferenceConfig(CustomConfig):
        DEVICE = args.device
        GPU_COUNT = args.g
        IMAGES_PER_GPU = args.batch
        DETECTION_MIN_CONFIDENCE = args.ci # Minimum probability value to accept a detected instance
        DETECTION_NMS_THRESHOLD = args.nms # Non-maximum suppression threshold for detection
        MAX_GT_INSTANCES = 1
        DETECTION_MAX_INSTANCES = 1

    inference_config = InferenceConfig()
    micro_model = modellib.MaskRCNN(mode="inference",
                            config=inference_config,
                            model_dir=DEFAULT_LOGS_DIR)

    micro_model_path = os.path.join(DEFAULT_LOGS_DIR, args.weights)

    print("Loading single-cell weights from ", micro_model_path)
    tf.keras.Model.load_weights(micro_model.keras_model, micro_model_path , by_name=True, skip_mismatch=True)

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(os.path.join(ROOT_DIR,"data"), "valid")
    dataset_val.prepare()

    image_norm_paths = []
    for filename in sorted(os.listdir(IMG_DIR_NORM)):
        if filename.endswith(".tif"):
            image_norm_paths.append(os.path.join(IMG_DIR_NORM, filename))

    for image_path in image_norm_paths:
        img_norm = skimage.io.imread(image_path)
        micro_results = micro_model.detect([img_norm], verbose=0)
        r = micro_results[0]
        mask = (r['masks'][:,:,0] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(MASK_DIR, os.path.basename(image_path)), mask)

    
    shutil.rmtree(IMG_DIR_NORM)
    apply_mask_to_imgs(IMG_DIR_CROP, MASK_DIR)
    


if __name__ == '__main__':
    main()
