import numpy as np
import os
import tifffile as tiff
import skimage
from aicsimageio import AICSImage
import argparse
import shutil


def czi_to_tiff(czi_folder,save_path,channels=None):
    """read czi files and save as tiff files"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, dirs, files in os.walk(czi_folder):
        for file in files:
            if file.endswith('.czi'):
                czi = AICSImage(os.path.join(root,file))
                image = czi.get_image_data("CZYX", S=0, T=0, Z=0)
                image = np.squeeze(image)
                image = np.moveaxis(image, 0, -1) #0,1,3
                if len(channels)==3:
                    new_image = np.stack((image[...,channels[0]],image[...,channels[1]],image[...,channels[2]]),axis=-1)
                elif len(channels)==2:
                    new_image = np.stack((image[...,channels[0]],image[...,channels[1]],np.zeros_like(image[...,0])),axis=-1)
                else:
                    new_image = np.stack((image[...,channels[0]],np.zeros_like(image[...,0]),np.zeros_like(image[...,0])),axis=-1)
                tiff.imwrite(os.path.join(save_path,file.replace('.czi','.tif')),new_image)

def ometifs_to_tifs(dir_path, save_path,channels=None):
    """converts ome-tifs to tifs"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(dir_path):
        if file.endswith('.ome.tif'):
            img = skimage.io.imread(os.path.join(dir_path, file))
            for i in range(img.shape[0]):
                skimage.io.imsave(os.path.join(save_path,file[:-8]+f'_{i}.tif'), img[i,...,channels])
                #skimage.io.imsave(os.path.join(save_path, file[:-8] + f'_{i}.tif'), img[i])

def tifs_to_tifs(dir_path,save_path,channels=None):
    """converts tifs to tifs with the correct channels"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(dir_path):
        if file.endswith('.tif'):
            img=skimage.io.imread(os.path.join(dir_path,file))
            skimage.io.imsave(os.path.join(save_path,file),img[...,channels])

def normalize_images(folder_path, save_path_norm, GRAYSCALE, clip_limit=0.02):
    """Normalize images and save them as grayscale"""
    if not os.path.exists(save_path_norm):
        os.makedirs(save_path_norm)
    for file in os.listdir(folder_path):
        if file.endswith('.tif'):
            original_img = skimage.io.imread(os.path.join(folder_path, file))
            if GRAYSCALE:
                float_gray_img = skimage.io.imread(os.path.join(folder_path, file), as_gray=True)
                float_gray_img_norm = skimage.exposure.equalize_adapthist(float_gray_img, clip_limit=clip_limit)
                img_norm = np.stack(((float_gray_img_norm * 255).astype(np.uint8),)*3, axis=-1)
            else:
                norm_channels = []
                for i in range(original_img.shape[2]):
                    if np.max(original_img[:,:,i]) > 0:
                        filt_channel = skimage.filters.median(original_img[:,:,i])
                        norm_channel = skimage.exposure.equalize_adapthist(filt_channel, clip_limit=clip_limit)
                        norm_channel = (norm_channel * 255).astype(np.uint8)
                    else:
                        norm_channel = original_img[:,:,i]
                    norm_channels.append(norm_channel)
                img_norm = np.stack(norm_channels, axis=-1)
            skimage.io.imsave(os.path.join(save_path_norm, os.path.basename(file)), img_norm)
    print(f'{len(os.listdir(save_path_norm))} images were normalized and saved.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing images for OL_MRCNN')
    parser.add_argument('--data', type=str, help='Path to the directory containing the images to be analyzed')
    parser.add_argument('--gray', action='store_true', help='Whether to convert images to grayscale')
    parser.add_argument('--type', type=str, default='tiff', help='Type of images to be analyzed (tiff, czi, ome-tif)')
    parser.add_argument('--clip', type=float, default=0.02, help='Clip limit for adaptive histogram equalization')
    parser.add_argument('--channels', type=str, default="[1,2,3]", help='Which channels to keep during preprocessing')

    args = parser.parse_args()
    data_path = args.data
    channels = [int(ch)-1 for ch in args.channels.strip("[]").split(",")[:-1]]

    if args.type == '.czi':
        output_path = data_path + '_tiff'
        czi_to_tiff(data_path, output_path,channels)
    elif args.type == '.ome.tif':
        output_path = data_path + '_tiff'
        ometifs_to_tifs(data_path, output_path,channels)
    else:
        output_path = data_path + '_tiff'
        tifs_to_tifs(data_path,output_path,channels)
        
    
    normalize_images(output_path, data_path+'_norm', args.gray, args.clip)
    shutil.rmtree(data_path+'_tiff')