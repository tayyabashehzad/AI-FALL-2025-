import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2

from common import config

random.seed(12345)
IMG_TYPES = ['*.jpg', '*.png']

def populate_train_list(orig_images_path, hazy_images_path):
    train_list = []
    val_list = []
    image_list_haze = []
    print(f"Searching for hazy images in: {hazy_images_path}")
    if not os.path.exists(hazy_images_path):
        print(f"Error: Hazy images directory NOT FOUND: {hazy_images_path}")
        return [], []
    if not os.path.exists(orig_images_path):
        print(f"Error: Original images directory NOT FOUND: {orig_images_path}")
        return [], []

    for img_type in IMG_TYPES:
        found = glob.glob(os.path.join(hazy_images_path, img_type))
        print(f"  Found {len(found)} files with pattern {img_type}")
        image_list_haze.extend(found)
    
    if len(image_list_haze) == 0:
        print(f"Warning: No hazy images found in {hazy_images_path} with patterns {IMG_TYPES}")
        return [], []
    
    mapping = {}
    for hz_image in image_list_haze:
        image_name = os.path.basename(hz_image)
        
        # Mapping logic:
        parts = image_name.split('_')
        gt_options = [
            image_name,
            parts[0] + os.path.splitext(image_name)[1], # Standard RESIDE
            image_name.replace('hazy', 'GT'),
            image_name.replace('hazy', 'clean')
        ]
        
        # NYU2 specific: NYU2_198_7_3.jpg -> 198.jpg or 198.png
        if len(parts) > 1:
            id_part = parts[1]
            gt_options.append(id_part + ".jpg")
            gt_options.append(id_part + ".png")
            gt_options.append(parts[0] + "_" + id_part + ".jpg")
            gt_options.append(parts[0] + "_" + id_part + ".png")
        
        found_gt = False
        for gt_name in gt_options:
            gt_path = os.path.join(orig_images_path, gt_name)
            if os.path.exists(gt_path):
                if gt_path not in mapping:
                    mapping[gt_path] = []
                mapping[gt_path].append(hz_image)
                found_gt = True
                break
        
        if not found_gt:
            # Optional: print first few failed matches to help debug
            if len(mapping) < 5:
                print(f"  Debug: Could not match {image_name}. Tried: {gt_options}")

    if len(mapping) == 0:
        print("Error: Could not match ANY hazy images to their clean/GT counterparts.")
        print(f"Check if image names in {hazy_images_path} correspond to {orig_images_path}")
        return [], []
    
    all_gt_keys = list(mapping.keys())
    random.shuffle(all_gt_keys)
    
    split_idx = int(len(all_gt_keys) * 0.9)
    train_keys = all_gt_keys[:split_idx]
    val_keys = all_gt_keys[split_idx:]

    for gt_path in train_keys:
        for hz_path in mapping[gt_path]:
            train_list.append([gt_path, hz_path])
            
    for gt_path in val_keys:
        for hz_path in mapping[gt_path]:
            val_list.append([gt_path, hz_path])

    return train_list, val_list


class dehazing_loader(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode='train'):
        self.mode = mode
        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = cv2.imread(data_orig_path)
        data_hazy = cv2.imread(data_hazy_path)
        
        if data_orig is None or data_hazy is None:
            return torch.zeros((3, config.height, config.width)), torch.zeros((3, config.height, config.width))

        data_orig = cv2.cvtColor(data_orig, cv2.COLOR_BGR2RGB)
        data_hazy = cv2.cvtColor(data_hazy, cv2.COLOR_BGR2RGB)

        data_orig = cv2.resize(data_orig, (config.height, config.width), cv2.INTER_LINEAR)
        data_hazy = cv2.resize(data_hazy, (config.height, config.width), cv2.INTER_LINEAR)
        
        # Data Augmentation (Flips)
        if self.mode == 'train' and random.random() > 0.5:
            data_orig = cv2.flip(data_orig, 1)
            data_hazy = cv2.flip(data_hazy, 1)

        data_orig = data_orig.astype(np.float32) / 255.0
        data_hazy = data_hazy.astype(np.float32) / 255.0
    
        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
