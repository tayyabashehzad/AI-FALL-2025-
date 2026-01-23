import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import numpy as np
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import glob

import net
from common import config

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def populate_test_list(gt_images_path, hazy_images_path):
    test_list = []
    print(f"Searching for test images in: {hazy_images_path}")
    
    image_list_haze = glob.glob(os.path.join(hazy_images_path, '*.jpg'))
    image_list_haze.extend(glob.glob(os.path.join(hazy_images_path, '*.png')))
    
    for hz_path in image_list_haze:
        image_name = os.path.basename(hz_path)
        
        # Try multiple GT filename matches
        parts = image_name.split('_')
        gt_options = [
            image_name, # Exact match
            parts[0] + os.path.splitext(image_name)[1], # RESIDE style
            image_name.replace('hazy', 'GT'), # NH-HAZE style
            image_name.replace('hazy', 'clean') 
        ]

        # NYU2 specific: NYU2_198_7_3.jpg -> 198.jpg or 198.png
        if len(parts) > 1:
            id_part = parts[1]
            gt_options.append(id_part + ".jpg")
            gt_options.append(id_part + ".png")
            gt_options.append(parts[0] + "_" + id_part + ".jpg")
            gt_options.append(parts[0] + "_" + id_part + ".png")
        
        found = False
        for gt_name in gt_options:
            gt_path = os.path.join(gt_images_path, gt_name)
            if os.path.exists(gt_path):
                test_list.append([gt_path, hz_path])
                found = True
                break
        
        if not found and len(test_list) < 5:
            print(f"  Debug: Could not find ground truth for {image_name}. Tried: {gt_options}")

    print(f"Total test samples found: {len(test_list)}")
    return test_list


def eval_index(gt_img, dehaze_img):
    gt_img = gt_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    dehaze_img = dehaze_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    psnr = peak_signal_noise_ratio(gt_img[0], dehaze_img[0])
    ssim = structural_similarity(gt_img[0], dehaze_img[0], channel_axis=-1, data_range=1)
    return psnr, ssim


def test(test_list, args):
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return 0, 0
    dehaze_net = net.dehaze_net().to(device)
    dehaze_net.load_state_dict(torch.load(args.model_path))
    dehaze_net.eval()

    print('\nStart Test!')
    total_psnr = 0
    total_ssim = 0
    for ite, (img_gt, img_haze) in enumerate(test_list):
        img_name = img_gt
        img_gt = cv2.imread(img_gt)
        img_haze = cv2.imread(img_haze)        
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_haze = cv2.cvtColor(img_haze, cv2.COLOR_BGR2RGB)
        if args.resize:
            img_gt = cv2.resize(img_gt, (config.height, config.width), cv2.INTER_LINEAR)
            img_haze = cv2.resize(img_haze, (config.height, config.width), cv2.INTER_LINEAR)
        img_gt = img_gt.transpose((2, 0, 1)) / 255.0
        img_haze = img_haze.transpose((2, 0, 1)) / 255.0
        img_gt = torch.from_numpy(img_gt).float().unsqueeze(0).to(device)
        img_haze = torch.from_numpy(img_haze).float().unsqueeze(0).to(device)
        with torch.no_grad():
            img_clean = dehaze_net(img_haze)
            img_clean = torch.clamp(img_clean, 0, 1)
        psnr, ssim = eval_index(img_gt, img_clean)
        total_psnr += psnr 
        total_ssim += ssim
        print('iter %d: ' % ite, os.path.split(img_name)[-1], ' PSNR: %.4f  SSIM %.4f' % (psnr, ssim))

        if args.save_image:
            torchvision.utils.save_image(torch.cat((img_haze, img_clean, img_gt), 0), 
                                     os.path.join(args.test_output_folder, os.path.basename(img_name)))

    num = len(test_list)
    if num == 0:
        print("Test over: No images to test.")
        return 0, 0
        
    total_psnr = total_psnr / num
    total_ssim = total_ssim / num
    print('Test numbers: %d' % num)
    print('Test Average PSNR: %.4f' % total_psnr)
    print('Test Average SSIM: %.4f' % total_ssim)

    return total_psnr, total_ssim


if __name__ == "__main__":

    # Auto-detect Environment
    on_colab = os.path.exists('/content/drive')
    
    if on_colab:
        default_base = os.getcwd() if '/content/drive' in os.getcwd() else "/content/drive/MyDrive/AOD-Net/baseline/"
    else:
        default_base = "./"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=os.path.join(default_base, "train_logs/snapshots/dehazer.pth"))
    parser.add_argument('--test_output_folder', type=str, default=os.path.join(default_base, "test_results/"))
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()
    for k, v in args.__dict__.items():
        print(k, ': ', v)

    if args.save_image and not os.path.exists(args.test_output_folder):
        os.makedirs(args.test_output_folder, exist_ok=True)
    
    # Update test paths
    if on_colab:
        # NH-Haze
        test_gt = "/content/drive/MyDrive/AOD_NET (Project)/baseline/NH-HAZE-Original-Images"
        test_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/NH-HAZE-Training-Images"
        # NYU2
        # test_gt = "/content/drive/MyDrive/AOD_NET (Project)/baseline/NYU2_Orginal_images/image/"
        # test_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/NYU2_Training_images/data/"
        # RESIDE
        # test_gt = "/content/drive/MyDrive/AOD_NET (Project)/baseline/RESIDE_Orginal_Images/clear/"
        # test_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/RESIDE_Training_Images/haze/"
        # I-HAZE
        # test_gt = "/content/drive/MyDrive/AOD_NET (Project)/baseline/I-HAZE_Orginal_Images/GT/"
        # test_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/I-HAZE_Training_Images/hazy/"
        # O-HAZE
        # test_gt = "/content/drive/MyDrive/AOD_NET (Project)/baseline/O-HAZE_Orginal_Images/GT/"
        # test_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/O-HAZE_Training_Images/hazy/"
    else:
        test_gt = '/content/drive/MyDrive/Reside_SOTS/indoor/clear/'
        test_hazy = '/content/drive/MyDrive/Reside_SOTS/indoor/hazy/'

    t1 = time.time()
    test_list = populate_test_list(test_gt, test_hazy)
    test_num = len(test_list)
    avg_psnr, avg_ssim = test(test_list, args)
    test_time = time.time() - t1
 
    print('\nTest Summary:')
    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Time: {test_time:.2f}s')
    print('Test over!')



 
