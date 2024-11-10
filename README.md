# Mask-R-CNN using Tensorflow2 for OL classification and segmentation

## OVERVIEW:
Model pipeline:
1. takes a .tif image, classifies the OLs and defines ROIs
2. crops the ROIs into a folder
3. applies segmentation to the cropped images

Efficiency:
- Detection: Recall=0.91
- Classification: Binary CE=0.76
- Segmentation: IoU=0.83

## LOCAL USE:
See the instructions set [here](https://drive.google.com/drive/folders/1PIstT451WQIOS59vtHqkq8PTD-xO0Gj_)