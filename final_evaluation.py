import torch
import torch.nn as nn
import torchvision
import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import lpips
import glob
import random

import dataloader
import net
from common import config

# Fixed seed for consistent 70-15-15 split
random.seed(12345)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_split_lists(orig_path, hazy_path):
    print(f"Splitting dataset into 70/15/15...")
    image_list_haze = glob.glob(os.path.join(hazy_path, "*.png"))
    image_list_haze.extend(glob.glob(os.path.join(hazy_path, "*.jpg")))

    mapping = {}
    for hz_image in image_list_haze:
        image_name = os.path.basename(hz_image)
        parts = image_name.split('_')
        gt_options = [
            image_name, 
            parts[0] + os.path.splitext(image_name)[1], 
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
            
        for gt_name in gt_options:
            gt_path = os.path.join(orig_path, gt_name)
            if os.path.exists(gt_path):
                if gt_path not in mapping: mapping[gt_path] = []
                mapping[gt_path].append(hz_image)
                break

    all_gt = list(mapping.keys())
    random.shuffle(all_gt)
    
    n = len(all_gt)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    splits = {
        'Training': all_gt[:train_end],
        'Validation': all_gt[train_end:val_end],
        'Testing': all_gt[val_end:]
    }

    final_lists = {}
    for name, gt_list in splits.items():
        pairs = []
        for gt in gt_list:
            for hz in mapping[gt]:
                pairs.append([gt, hz])
        final_lists[name] = pairs
        print(f"  {name}: {len(pairs)} image pairs")
    
    return final_lists

def run_evaluation(name, image_pairs, dehaze_net, loss_fn_vgg, output_dir):
    print(f"\nEvaluating {name} Set...")
    metrics = {'psnr': [], 'ssim': [], 'mse': [], 'lpips': []}
    viz_images = []

    for i, (gt_path, hz_path) in enumerate(image_pairs):
        # Load and process images
        img_gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        img_hz = cv2.cvtColor(cv2.imread(hz_path), cv2.COLOR_BGR2RGB)
        
        # Resize to study dimensions
        img_gt_res = cv2.resize(img_gt, (config.height, config.width))
        img_hz_res = cv2.resize(img_hz, (config.height, config.width))
        
        # CPU versions for math metrics (numpy)
        gt_norm = img_gt_res / 255.0
        hz_norm = img_hz_res / 255.0

        # GPU versions for Model and LPIPS
        t_hz = torch.from_numpy(hz_norm.transpose(2,0,1)).float().unsqueeze(0).to(device)
        t_gt = torch.from_numpy(gt_norm.transpose(2,0,1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            t_pred = dehaze_net(t_hz)
            t_pred = torch.clamp(t_pred, 0, 1)
            
            # LPIPS calculation (requires [-1, 1] range)
            lp_dist = loss_fn_vgg(t_pred * 2 - 1, t_gt * 2 - 1).item()
        
        pred_norm = t_pred.cpu().numpy()[0].transpose(1, 2, 0)

        # Basic Metrics
        metrics['psnr'].append(peak_signal_noise_ratio(gt_norm, pred_norm))
        metrics['ssim'].append(structural_similarity(gt_norm, pred_norm, channel_axis=-1, data_range=1))
        metrics['mse'].append(mean_squared_error(gt_norm, pred_norm))
        metrics['lpips'].append(lp_dist)

        # Collect 10 images for visualization
        if len(viz_images) < 10:
            viz_images.append((img_hz_res, (pred_norm * 255).astype(np.uint8), img_gt_res))

    # Print Results
    avg_results = {k: np.mean(v) for k, v in metrics.items()}
    print(f"  MSE: {avg_results['mse']:.6f} | PSNR: {avg_results['psnr']:.4f} | SSIM: {avg_results['ssim']:.4f} | LPIPS: {avg_results['lpips']:.4f}")

    # Plot Grid (10x3)
    if viz_images:
        fig, axes = plt.subplots(3, 10, figsize=(20, 7))
        for j in range(len(viz_images)):
            titles = ["Hazy Input", "Dehazed (Pred)", "Ground Truth"]
            for r in range(3):
                axes[r, j].imshow(viz_images[j][r])
                axes[r, j].axis('off')
                if j == 0: axes[r, j].set_ylabel(titles[r], rotation=0, labelpad=40, size='large', weight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_visualization.png"), bbox_inches='tight', dpi=150)
        plt.close()

    return avg_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    on_colab = os.path.exists('/content/drive')
    default_base = os.getcwd() if on_colab else "./"
    
    parser.add_argument('--model_path', type=str, default=os.path.join(default_base, "train_logs/snapshots/dehazer.pth"))
    # NH-Haze
    parser.add_argument('--orig_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/NH-HAZE-Original-Images")
    parser.add_argument('--hazy_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/NH-HAZE-Training-Images")
    # NYU2
    # parser.add_argument('--orig_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/NYU2_Orginal_images/image/")
    # parser.add_argument('--hazy_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/NYU2_Training_images/data/")
    # RESIDE
    # parser.add_argument('--orig_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/RESIDE_Orginal_Images/clear/")
    # parser.add_argument('--hazy_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/RESIDE_Training_Images/haze/")
    # IHAZE
    # parser.add_argument('--orig_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/I-HAZE_Orginal_Images/GT/")
    # parser.add_argument('--hazy_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/I-HAZE_Training_Images/hazy/")
    # OHAZE
    # parser.add_argument('--orig_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/O-HAZE_Orginal_Images/GT/")
    # parser.add_argument('--hazy_path', type=str, default="/content/drive/MyDrive/AOD_NET (Project)/baseline/O-HAZE_Training_Images/hazy/")
    parser.add_argument('--output_dir', type=str, default=os.path.join(default_base, "evaluation_report"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    dehaze_net = net.dehaze_net().to(device)
    dehaze_net.load_state_dict(torch.load(args.model_path, map_location=device))
    dehaze_net.eval()

    # 2. Load LPIPS
    print("Loading LPIPS VGG model...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # 3. Get Splits
    data_splits = get_split_lists(args.orig_path, args.hazy_path)

    # 4. Evaluate and Save
    summary_file = os.path.join(args.output_dir, "quantitative_results.txt")
    with open(summary_file, 'w') as f:
        f.write("AOD-Net Formal Evaluation Report\n")
        f.write("="*30 + "\n")
        
        for name, pairs in data_splits.items():
            results = run_evaluation(name, pairs, dehaze_net, loss_fn_vgg, args.output_dir)
            f.write(f"\n{name} Set Results:\n")
            for k, v in results.items():
                f.write(f"  {k.upper()}: {v:.6f}\n")

    print(f"\nEvaluation Complete! Reports and plots saved in: {args.output_dir}")
