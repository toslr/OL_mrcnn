import numpy as np
import os
import cv2
import pandas as pd
import argparse
import skimage
from aicsimageio import AICSImage

def detect_overlap(bboxes):
    """List overlaps between the bounding boxes."""
    overlaps = []
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            x1, y1, w1, h1 = bboxes[i]
            x2, y2, w2, h2 = bboxes[j]
            if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
                overlaps.append((i, j))
    return overlaps

def nms_suppression_multi(results,threshold):
    """Apply non-maximum suppression to avoid multiple detections of the same object"""
    r = results[0]
    remove_index = np.array([0]*len(r['rois'])).astype(bool)
    for i in range(0,len(r['rois'])):
        for j in range(i+1,len(r['rois'])):
            if remove_index[i] == 0 and remove_index[j] == 0:
                iou = np.logical_and(r['masks'][:,:,i].astype(bool),r['masks'][:,:,j].astype(bool)).sum() / np.logical_or(r['masks'][:,:,i].astype(bool),r['masks'][:,:,j].astype(bool)).sum()
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

def crop_from_csv(csv_results,img_dir,res_dir):
    """Crop ROIs in images from csv results"""
    print(img_dir)
    results = pd.read_csv(csv_results)
    results['detection_id'] = [i+1 for i in range(len(results))]
    results.to_csv(csv_results,index=False)
    image_names = sorted(results['image_name'].unique())
    for file in os.listdir(res_dir):
        if file.endswith('.tif'):
            os.remove(os.path.join(res_dir,file))
    for image_name in image_names:
        cziCheck = False
        detection_ids = results[results['image_name']==image_name]['detection_id'].values
        classes = results[results['image_name']==image_name]['class'].values
        str_rois = results[results['image_name']==image_name]['bbox'].values
        rois =[[int(coord) for coord in str_roi.strip('[]').split()] for str_roi in str_rois]
        if image_name.endswith('.czi'):
            cziCheck = True
            czi = AICSImage(os.path.join(img_dir,image_name))
            image = czi.get_image_data("CZYX", S=0, T=0, Z=0)
            image = np.squeeze(image)
            image = np.moveaxis(image, 0, -1)
        else:
            image = skimage.io.imread(os.path.join(img_dir,image_name))
        if cziCheck:
            for i in range(len(detection_ids)):
                skimage.io.imsave(os.path.join(res_dir,os.path.basename(image_name)[:-4] + f'_{i:04d}_' + classes[i] + '.czi'),image[rois[i][0]:rois[i][2],rois[i][1]:rois[i][3]])
        else:
            for i in range(len(detection_ids)):
                skimage.io.imsave(os.path.join(res_dir,os.path.basename(image_name)[:-4] + f'_{i:04d}_' + classes[i] + '.tif'),image[rois[i][0]:rois[i][2],rois[i][1]:rois[i][3]])


def crop_from_results(image_path,image,img_res_dir,results,res_list,class_names,modify_results=False,save_images=False):
    """Crop ROIs from image using the results, and correct ROIs to display only 1 object in the cropped region."""
    overlaps = detect_overlap(results[0]['rois'])
    if len(overlaps) > 0:
        print('Warning: Overlapping bounding boxes detected. Applying non-maximum suppression.')
        results = nms_suppression_multi(results,0.5)
    r = results[0]
    if not modify_results:
        for i in range(r['masks'].shape[2]):
            class_name = class_names[r['class_ids'][i]]
            res_list.append([os.path.basename(image_path), len(res_list), class_names[r['class_ids'][i]], r['scores'][i], r['rois'][i]])
            cropped_img = image[r['rois'][i][0]:r['rois'][i][2], r['rois'][i][1]:r['rois'][i][3]]
            if save_images:
                cv2.imwrite(os.path.join(img_res_dir, os.path.basename(image_path)[:-3] + f'{i:04d}_' + class_name + '.tif'), cropped_img)
    
    else:
        for i in range(r['masks'].shape[2]):
            class_name = class_names[r['class_ids'][i]]
            res_list.append([os.path.basename(image_path), len(res_list), class_names[r['class_ids'][i]], r['scores'][i], r['rois'][i]])
            overlaps_with_i = [pair[1] if pair[0]==i else pair[0] for pair in overlaps if pair[0]==i or pair[1]==i]
            modified_image = image.copy()
            
            bbox_i_mask = np.zeros_like(r['masks'][:,:,0])
            bbox_i_mask[r['rois'][i][0]:r['rois'][i][2], r['rois'][i][1]:r['rois'][i][3]] = 1
            to_erase = np.zeros_like(r['masks'][:,:,0])
            for j in overlaps_with_i:
                to_erase += np.logical_and(np.logical_and(bbox_i_mask,1-r['masks'][:,:,i].astype(bool)), r['masks'][:,:,j].astype(bool)) # erase overlapping between bbox i and mask j but not mask i
            modified_image[to_erase.astype(bool)] = 0
            cropped_img = modified_image[r['rois'][i][0]:r['rois'][i][2], r['rois'][i][1]:r['rois'][i][3]]
            if save_images:
                cv2.imwrite(os.path.join(img_res_dir, os.path.basename(image_path)[:-3] + f'{i:04d}_' + class_name + '.tif'), cropped_img)
    return res_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to the csv file containing the results')
    parser.add_argument('--img_dir', type=str, help='Path to the directory containing the images')
    parser.add_argument('--res_dir', type=str, help='Path to the directory where the cropped images will be saved')
    args = parser.parse_args()
    crop_from_csv(args.csv, args.img_dir[:-5], args.res_dir)