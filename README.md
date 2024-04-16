# Mask-R-CNN using Tensorflow2 for OL classification and segmentation

## OVERVIEW:
Model pipeline:
1. takes a .tif image, classifies the OLs and defines ROIs
2. crops the ROIs into a folder
3. applies segmentation to the cropped images

Output: in results/'taskname'/ produces results.csv and one folder per image

Efficiency:
- Detection: Recall=0.91
- Classification: Binary CE=0.76
- Segmentation: IoU=0.77

## SETUP FOR IMAGE CROPPER:
Tested under MacOS but should be working under any OS.
1. create conda env: conda env create -f environment.yml
2. Download the weight files here: https://drive.google.com/drive/folders/1PIstT451WQIOS59vtHqkq8PTD-xO0Gj_?usp=sharing and move them to the logs/ directory
3. in image_cropper.ipynb, change parameters and run the script
