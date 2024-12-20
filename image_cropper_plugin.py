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
from preprocessing import czi_to_tiff, ometifs_to_tifs, normalize_images
from postprocessing import nms_suppression_multi, crop_from_csv, crop_from_results

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
    parser.add_argument('--name', type=str, default='results0', help='Name of results directory')
    parser.add_argument('--data', type=str, default='data/test/imgs', help='Directory containing test images')
    parser.add_argument('--gray', action='store_true', help='Whether to convert images to grayscale')
    parser.add_argument('--edit', action='store_true', help='Whether to visualize and edit the results')
    parser.add_argument('--multiclass', action='store_true', help='Whether to use the multiclass model')
    parser.add_argument('--prep', action='store_true', help='Whether to preprocess the folder of images')
    parser.add_argument('--save', action='store_true', default=True, help='Whether to save the crops')
    args = parser.parse_args()

    IMG_DIR = args.data
    GRAYSCALE = args.gray

    if args.prep:
        for file in os.listdir(IMG_DIR):
            if file.endswith('.ome.tif'):
                ometifs_to_tifs(IMG_DIR,IMG_DIR+'_tiff')
                normalize_images(IMG_DIR+'_tiff',IMG_DIR+'_norm', GRAYSCALE)
                IMG_DIR = IMG_DIR+'_norm'
                break
            if file.endswith('.czi'):
                czi_to_tiff(IMG_DIR,IMG_DIR+'_tiff')
                normalize_images(IMG_DIR+'_tiff',IMG_DIR+'_norm', GRAYSCALE)
                IMG_DIR = IMG_DIR+'_norm'
                break
            if file.endswith('.tiff'):
                normalize_images(IMG_DIR,IMG_DIR+'_norm', GRAYSCALE)
                IMG_DIR = IMG_DIR+'_norm'
                break
    
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

    inference_config = InferenceConfig()
    macro_model = modellib.MaskRCNN(mode="inference",
                                config=inference_config,
                                model_dir=DEFAULT_LOGS_DIR)

    macro_model_path = os.path.join(DEFAULT_LOGS_DIR, args.weights)

    print("Loading macro weights from ", macro_model_path)
    tf.keras.Model.load_weights(macro_model.keras_model, macro_model_path , by_name=True, skip_mismatch=True)
    if IMG_DIR.endswith('/'):
        IMG_DIR = IMG_DIR[:-1]
    if not IMG_DIR.endswith('_norm'):
        print('Selecting normalized images')
        IMG_DIR = IMG_DIR+'_norm'
    RESULTS_DIR = os.path.join(os.path.dirname(IMG_DIR), args.name)
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
        img_norm = skimage.io.imread(image_path)

        macro_results = macro_model.detect([img_norm], verbose=0)
        macro_results = nms_suppression_multi(macro_results, args.nms)
        r = macro_results[0]
        for i in range(len(r['rois'])):
            res_list.append([os.path.basename(image_path), 
                             len(res_list)+1, 
                             dataset_val.class_names[r['class_ids'][i]], 
                             r['scores'][i],
                             r['rois'][i]])

    res_df = pd.DataFrame(res_list, columns=['image_name', 'detection_id', 'class', 'score', 'bbox'])
    res_df.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)
    crop_from_csv(os.path.join(RESULTS_DIR, 'results.csv'), IMG_DIR, os.path.join(RESULTS_DIR, 'crops'))


if __name__ == '__main__':
    main()