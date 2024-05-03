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

## LOCAL USE:
Open the index.html in a web browser and follow the instructions.
