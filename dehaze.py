import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import glob
from common import config

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dehaze_piece_image(dehaze_net, data_haze, clean_image, i1, i2, j1, j2, pad):
    haze_piece = data_haze[:, :, i1:i2+2*pad, j1:j2+2*pad]
    pred_piece = dehaze_net(haze_piece).cpu().detach().numpy()[0]
    if pad == 0:
        clean_image[:, i1:i2, j1:j2] = pred_piece[:, :, :]
    else:
        clean_image[:, i1:i2, j1:j2] = pred_piece[:, pad:-pad, pad:-pad]
    return clean_image


def crop_splice_image(dehaze_net, data_haze, args):
    h_piece = args.h_piece
    w_piece = args.w_piece
    pad = args.pad
    
    H, W = data_haze.size()[2], data_haze.size()[3]
    if pad > 0:
        pad_haze = torch.zeros(1, 3, H+2*pad, W+2*pad)
        pad_haze[:, :, pad:-pad, pad:-pad] = data_haze
        data_haze = pad_haze.to(device)
    
    clean_image = np.zeros((3, H, W))
    h_count = H // h_piece
    w_count = W // w_piece
    h_left = H % h_piece
    w_left = W % w_piece
    for i in range(h_count):
        for j in range(w_count):
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i*h_piece, (i+1)*h_piece, j*w_piece, (j+1)*w_piece, pad)
        if w_left > 0:
            j = w_count
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i*h_piece, (i+1)*h_piece, j*w_piece, W+1, pad)
    if h_left > 0:
        i = h_count
        for j in range(w_count):
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i*h_piece, H+1, j*w_piece, (j+1)*w_piece, pad)
        if h_left > 0 and w_left > 0:
            j = w_count
            clean_image = dehaze_piece_image(dehaze_net, data_haze, clean_image, i*h_piece, H+1, j*w_piece, W+1, pad)
    return clean_image


def post_process(image):
    for k in range(image.shape[-1]):
        min_pixel = np.min(image[:, :, k])
        max_pixel = np.max(image[:, :, k])
        if max_pixel > min_pixel:
            image[:, :, k] = (image[:, :, k] - min_pixel) / (max_pixel - min_pixel)
    factor = 0.5
    image = np.clip(image/factor, 0, 1) 
    return image


def dehaze_image(image_path, args, dehaze_net): 
    data_haze = cv2.imread(image_path)
    if data_haze is None: return
    data_haze = cv2.cvtColor(data_haze, cv2.COLOR_BGR2RGB)
    data_haze = data_haze.transpose((2, 0, 1)) / 255.0
    
    data_haze = torch.from_numpy(data_haze).float()
    data_haze = data_haze.to(device).unsqueeze(0)

    with torch.no_grad():
        if not args.crop:
            clean_image = dehaze_net(data_haze)
            clean_image = clean_image.cpu().detach().numpy()[0] 
            save_path = os.path.join(args.output_dir, os.path.basename(image_path))
        else:
            clean_image = crop_splice_image(dehaze_net, data_haze, args)
            save_path = os.path.join(args.output_dir, 'smooth_' + os.path.basename(image_path))
            
    clean_image = clean_image.transpose((1, 2, 0)) 
    if args.post_process:
        clean_image = post_process(clean_image)
    else:
        clean_image = np.clip(clean_image, 0, 1)
        
    show_clean_image = np.uint8(clean_image * 255)
    show_clean_image = cv2.cvtColor(show_clean_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, show_clean_image)


if __name__ == '__main__':
    # Auto-detect Environment
    on_colab = os.path.exists('/content/drive')
    
    if on_colab:
        default_base = os.getcwd() if '/content/drive' in os.getcwd() else "/content/drive/MyDrive/AOD-Net/baseline/"
        default_test = "/content/drive/MyDrive/nh-haze-hazy-images"
    else:
        default_base = "./"
        default_test = "../test_images"

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default=default_test)
    parser.add_argument('--model_path', type=str, default=os.path.join(default_base, 'train_logs/snapshots/dehazer.pth'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(default_base, 'demo_results'))
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--h_piece', type=int, default=config.height)
    parser.add_argument('--w_piece', type=int, default=config.width)
    parser.add_argument('--pad', type=int, default=3)
    parser.add_argument('--post_process', action='store_true', default=False)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)

    dehaze_net = net.dehaze_net().to(device)
    dehaze_net.load_state_dict(torch.load(args.model_path, map_location=device))
    dehaze_net.eval()
    
    test_list = glob.glob(os.path.join(args.test_dir, '*'))
    for image in test_list:
        if os.path.isdir(image): continue
        t1 = time.time()
        dehaze_image(image, args, dehaze_net)
        print(f"{os.path.basename(image)} done! Time: {time.time()-t1:.2f}s")
