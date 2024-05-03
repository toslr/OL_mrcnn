import tensorflow as tf
print("Tensorflow", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import tifffile as tiff

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from custom_multi import nms_suppression_multi
from custom_multi import CustomConfig as CustomConfigMulti
from custom_multi import CustomDataset as CustomDatasetMulti
from custom import CustomConfig, CustomDataset


# %%
# Creates a session with device placement logs
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(sess)

# %% [markdown]
# ## Parameters

# %%
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
gpu_count_macro = 1
num_img_per_gpu_macro = 1
min_confidence_macro = 0.7
nms_threshold_macro = 0.3
nms_multiclass_macro = 0.3
gpu_count_micro = 1
num_img_per_gpu_micro = 1
min_confidence_micro = 0.7
nms_threshold_micro = 0.3
MACRO_MODEL_SUBPATH = 'multicell20240319T2245/mask_rcnn_multicell_0050.h5'
MICRO_MODEL_SUBPATH = 'cell20240302T1503/mask_rcnn_cell_0050.h5'
RESULTS_NAME = 'test_full_pipeline'
TEST_DIR = '/Users/tom/Desktop/Stanford/RA/OligodendroSight/OL_mrcnn/data/test/imgs'

# %%
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", RESULTS_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# %% [markdown]
# ## Classification and cropping

# %%
class InferenceConfigMulti(CustomConfigMulti):
    GPU_COUNT = gpu_count_macro
    IMAGES_PER_GPU = num_img_per_gpu_macro
    DETECTION_MIN_CONFIDENCE = min_confidence_macro #Minimum probability value to accept a detected instance
    DETECTION_NMS_THRESHOLD = nms_threshold_macro # Non-maximum suppression threshold for detection

inference_config_multi = InferenceConfigMulti()

macro_model = modellib.MaskRCNN(mode="inference",
                            config=inference_config_multi,
                            model_dir=DEFAULT_LOGS_DIR)

macro_model_path = os.path.join(DEFAULT_LOGS_DIR, MACRO_MODEL_SUBPATH)
print("Loading macro weights from ", macro_model_path)
#model.load_weights(model_path, by_name=True) #deprecated on Python 3.9
tf.keras.Model.load_weights(macro_model.keras_model, macro_model_path , by_name=True, skip_mismatch=True)


# %%
# Validation dataset
dataset_val = CustomDatasetMulti()
dataset_val.load_custom("/Users/tom/Desktop/Stanford/RA/OligodendroSight/OL_mrcnn/data", "valid")
dataset_val.prepare()

# %%
image_paths = []
for filename in os.listdir(TEST_DIR):
    if filename.endswith(".tif"):
        image_paths.append(os.path.join(TEST_DIR, filename))

res_list = []
for image_path in image_paths:
    print(os.path.basename(image_path))
    image = skimage.io.imread(image_path)
    macro_results = macro_model.detect([image], verbose=0)
    macro_results = nms_suppression_multi(macro_results, nms_multiclass_macro)
    r = macro_results[0]
    img_res_dir = os.path.join(RESULTS_DIR, os.path.basename(image_path)[:-4])
    os.makedirs(img_res_dir, exist_ok=True)
    for i in range(r['masks'].shape[2]):
        class_name = dataset_val.class_names[r['class_ids'][i]]
        res_list.append([os.path.basename(image_path), i, dataset_val.class_names[r['class_ids'][i]], r['scores'][i], r['rois'][i]])
        cropped_img = image[r['rois'][i][0]:r['rois'][i][2], r['rois'][i][1]:r['rois'][i][3]]
        cv2.imwrite(os.path.join(img_res_dir, f'{i:04d}_' + class_name + '.tif'), cropped_img)
    

res_df = pd.DataFrame(res_list, columns=['image_name', 'detection_id', 'class', 'score', 'bbox'])
res_df.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)

# %% [markdown]
# ## Segmentation

# %%
class InferenceConfigSingle(CustomConfig):
    GPU_COUNT = gpu_count_micro
    IMAGES_PER_GPU = num_img_per_gpu_micro
    DETECTION_MIN_CONFIDENCE = min_confidence_micro #Minimum probability value to accept a detected instance
    DETECTION_NMS_THRESHOLD = nms_threshold_micro # Non-maximum suppression threshold for detection

inference_config_single = InferenceConfigSingle()

micro_model = modellib.MaskRCNN(mode="inference",
                            config=inference_config_single,
                            model_dir=DEFAULT_LOGS_DIR)

micro_model_path = os.path.join(DEFAULT_LOGS_DIR, MICRO_MODEL_SUBPATH)

print("Loading micro weights from ", micro_model_path)
tf.keras.Model.load_weights(micro_model.keras_model, micro_model_path , by_name=True, skip_mismatch=True)

# %%
for i in range(len(res_df)):
    img_path = os.path.join(RESULTS_DIR, res_df.iloc[i]['image_name'][:-4], f"{res_df.iloc[i]['detection_id']:04d}_" + res_df.iloc[i]['class'] + '.tif')
    img = skimage.io.imread(img_path)
    micro_results = micro_model.detect([img], verbose=1)
    r = micro_results[0]
    mask = (r['masks'][:,:,0]*255).astype(np.uint8)
    cv2.imwrite(os.path.join(RESULTS_DIR, res_df.iloc[i]['image_name'][:-4], f'{i:04d}_' + res_df.iloc[i]['class'] + '_mask.tif'), mask)


