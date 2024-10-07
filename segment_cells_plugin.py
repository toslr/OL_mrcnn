import tensorflow as tf

# Creates a session with device placement logs
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

import warnings
warnings.filterwarnings('ignore')
import os
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

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from preprocessing import czi_to_tiff, ometifs_to_tifs, normalize_images
from postprocessing import nms_suppression_multi, crop_from_csv, crop_from_results

def main():
    print("Command line arguments:", ' '.join(sys.argv))
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
    parser.add_argument('--name', type=str, default='results0', help='Name of results directory')
    parser.add_argument('--data', type=str, default='data/test/imgs', help='Directory containing test images')
    parser.add_argument('--gray', action='store_true', help='Whether to convert images to grayscale')
    parser.add_argument('--multiclass', action='store_true', help='Whether to use the multiclass model')
    parser.add_argument('--save', action='store_true', default=True, help='Whether to save the crops')

    args = parser.parse_args()

    IMG_DIR = args.data
    GRAYSCALE = args.gray

    if args.multiclass:
        from custom_multi import CustomConfig, CustomDataset
    else:
        from custom import CustomConfig, CustomDataset
    
    class InferenceConfig(CustomConfig):
        GPU_COUNT = args.g
        IMAGES_PER_GPU = args.batch
        DETECTION_MIN_CONFIDENCE = args.ci
        DETECTION_NMS_THRESHOLD = args.nms
        NAME = args.name
        DEVICE = args.device
        NUM_CLASSES = 1 + 1
    
    inference_config = InferenceConfig()
    micro_model = modellib.MaskRCNN(mode="inference", 
                                    config=inference_config, 
                                    model_dir=DEFAULT_LOGS_DIR)
    micro_model_path = os.path.join(DEFAULT_LOGS_DIR, args.weights)
    print("Loading micro weights from ", micro_model_path)
    tf.keras.Model.load_weights(micro_model.keras_model, micro_model_path , by_name=True, skip_mismatch=True)
    RESULTS_NAME = args.name
    RESULTS_DIR = os.path.join(ROOT_DIR, "results", RESULTS_NAME)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(os.path.join(ROOT_DIR,"data"), "valid")
    dataset_val.prepare()

    image_paths = []
    for filename in sorted(os.listdir(IMG_DIR)):
        if filename.endswith(".tif"):
            image_paths.append(os.path.join(IMG_DIR, filename))

    res_list = []
    for image_path in image_paths:
        print(image_path)
        original_img = skimage.io.imread(image_path)
        img_norm = original_img

        micro_results = micro_model.detect([img_norm], verbose=0)
        micro_results = nms_suppression_multi(micro_results, args.nms)
        r = micro_results[0]
        for i in range(len(r['rois'])):
            res_list.append([os.path.basename(image_path), 
                             len(res_list)+1, 
                             dataset_val.class_names[r['class_ids'][i]], 
                             r['scores'][i],
                             r['rois'][i]])
            res_df = pd.DataFrame(res_list, columns=['image_name', 'detection_id', 'class', 'score', 'bbox'])
    res_df.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)
    crop_from_csv(os.path.join(RESULTS_DIR, 'results.csv'), IMG_DIR, RESULTS_DIR)

if __name__ == '__main__':
    main()

