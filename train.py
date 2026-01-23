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

import dataloader
import net
from common import config

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, nn.Conv2d):  # proposed way
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def eval_index(gt_img, dehaze_img):
    gt_img = gt_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    dehaze_img = dehaze_img.cpu().detach().numpy().transpose((0, 2, 3, 1))
    psnr, ssim = 0, 0
    N = gt_img.shape[0]
    for i in range(N):
        psnr += peak_signal_noise_ratio(gt_img[i], dehaze_img[i])
        ssim += structural_similarity(gt_img[i], dehaze_img[i], channel_axis=-1, data_range=1)
    return psnr/N, ssim/N


import torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([torch.exp(-(torch.FloatTensor([x - window_size // 2])**2) / (2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean() if self.size_average else 1 - ssim_map

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super(HybridLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.alpha = alpha
    def forward(self, pred, target):
        return self.alpha * self.ssim(pred, target) + (1 - self.alpha) * self.l1(pred, target)

def train(args, dehaze_net):
    if args.model_trained:
        dehaze_net.load_state_dict(torch.load(args.model_path))
        print('\nModel loaded without train!')
    else:
        print('\nStart train with Enhanced Architecture!')
        dehaze_net.apply(weights_init)

        train_dataset = dataloader.dehazing_loader(args.orig_images_path,
                                                 args.hazy_images_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                            shuffle=True, num_workers=args.num_workers, pin_memory=True)

        if args.loss_func == 'hybrid':
            criterion = HybridLoss().to(device)
        elif args.loss_func == 'l1':
            criterion = nn.SmoothL1Loss().to(device)
        elif args.loss_func == 'l2':
            criterion = nn.MSELoss().to(device)
        else:
            criterion = HybridLoss().to(device) # Default to hybrid
            print('loss_func %s not supported, using Hybrid Loss' % args.loss_func)
        
        optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

        dehaze_net.train()

        for epoch in range(args.num_epochs):
            print('lr:%.6f' % schedule.get_last_lr()[0])
            total_loss = 0
            for iteration, (img_orig, img_haze) in enumerate(train_loader):
                img_orig = img_orig.to(device)
                img_haze = img_haze.to(device)

                clean_image = dehaze_net(img_haze)
                loss = criterion(clean_image, img_orig)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), args.grad_clip_norm)
                optimizer.step()

                if ((iteration+1) % args.display_iter) == 0:
                    print("Loss at epoch", epoch+1, "| iteration", iteration+1, ":%.8f" % loss.item())
                total_loss += loss.item()
                if ((iteration+1) % args.snapshot_iter) == 0:
                    save_model = dehaze_net.module.state_dict() if args.multi_gpu else dehaze_net.state_dict()
                    torch.save(save_model, args.snapshots_folder + "Epoch" + str(epoch) + '.pth')
            print('Average loss at epoch', epoch+1, ":%.8f" % (total_loss/len(train_loader)))
            schedule.step()  # adjust learning rate
            save_model = dehaze_net.module.state_dict() if args.multi_gpu else dehaze_net.state_dict()
            torch.save(save_model, os.path.join(args.snapshots_folder, "dehazer.pth"))
    
    # valid(args, dehaze_net)

# Validation Stage
def valid(args, dehaze_net):
    print('\nStart validation!')
    val_dataset = dataloader.dehazing_loader(args.orig_images_path,
                                            args.hazy_images_path, mode="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)  # shuffle=True

    dehaze_net.eval()
    total_psnr = 0
    total_ssim = 0
    for iter_val, (img_orig, img_haze) in enumerate(val_loader):
        img_orig = img_orig.to(device)
        img_haze = img_haze.to(device)
        with torch.no_grad():
            clean_image = dehaze_net(img_haze)
            clean_image = torch.clamp(clean_image, 0, 1)

        psnr, ssim = eval_index(img_orig, clean_image)
        total_psnr += psnr 
        total_ssim += ssim
        print('Batch %d - Validate PSNR: %.4f' % (iter_val, psnr))
        print('Batch %d - Validate SSIM: %.4f' % (iter_val, ssim))
        
        # permute [2,1,0] means convert BGR to RGB to display by torchvision
        if not args.valid_not_save:
            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                         os.path.join(args.sample_output_folder, str(iter_val+1)+".jpg"))

    total_psnr = total_psnr / len(val_dataset) * args.val_batch_size
    total_ssim = total_ssim / len(val_dataset) * args.val_batch_size
    print('Validate PSNR: %.4f' % total_psnr)
    print('Validate SSIM: %.4f' % total_ssim)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--dataset', type=str, default='ITS', choices=['ITS', 'OTS'])
    lr = parser.add_argument_group(title='Learning rate')
    lr.add_argument('--init_lr', type=float, default=0.0001)
    lr.add_argument('--milestones', nargs='+', type=int, default=[4, 7])
    lr.add_argument('--gamma', type=float, default=0.1)     
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--loss_func', default='l1', help='l1|l2')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    
    # Auto-detect Environment
    on_colab = os.path.exists('/content/drive')
    
    if on_colab:
        # Detect where we are on Drive (handles spaces and custom names)
        curr_dir = os.getcwd()
        if '/content/drive' in curr_dir:
            default_base = curr_dir
        else:
            default_base = "/content/drive/MyDrive/AOD-Net/baseline/"
    else:
        default_base = "./"
        
    parser.add_argument('--snapshots_folder', type=str, default=os.path.join(default_base, "train_logs/snapshots/"))
    parser.add_argument('--sample_output_folder', type=str, default=os.path.join(default_base, "train_logs/samples/"))
    parser.add_argument('--model_path', type=str, default=os.path.join(default_base, "train_logs/snapshots/dehazer.pth"))
    
    parser.add_argument('--model_trained', action='store_true')
    parser.add_argument('--valid_not_save', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')

    # Data paths - Now added as arguments to allow override via CLI
    if on_colab:
        # NH-Haze
        def_orig = "/content/drive/MyDrive/AOD_NET (Project)/baseline/NH-HAZE-Original-Images"
        def_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/NH-HAZE-Training-Images"
        # NYU2
        # def_orig = "/content/drive/MyDrive/AOD_NET (Project)/baseline/NYU2_Orginal_images/image/"
        # def_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/NYU2_Training_images/data/"
        # RESIDE
        # def_orig = "/content/drive/MyDrive/AOD_NET (Project)/baseline/RESIDE_Orginal_Images/clear/"
        # def_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/RESIDE_Training_Images/haze/"
        # I-Haze
        # def_orig = "/content/drive/MyDrive/AOD_NET (Project)/baseline/I-HAZE_Orginal_Images/GT/"
        # def_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/I-HAZE_Training_Images/hazy/"
        # O-Haze
        # def_orig = "/content/drive/MyDrive/AOD_NET (Project)/baseline/O-HAZE_Orginal_Images/GT/"
        # def_hazy = "/content/drive/MyDrive/AOD_NET (Project)/baseline/O-HAZE_Training_Images/hazy/"
    else:
        def_orig = "./data/clear/"
        def_hazy = "./data/hazy/"

    parser.add_argument('--orig_images_path', type=str, default=def_orig)
    parser.add_argument('--hazy_images_path', type=str, default=def_hazy)

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(k, ': ', v)
    
    if not os.path.exists(args.snapshots_folder):
        os.system('mkdir -p %s' % args.snapshots_folder)
    if not os.path.exists(args.sample_output_folder):
        os.system('mkdir -p %s' % args.sample_output_folder)

    t1 = time.time()
    dehaze_net = net.dehaze_net().to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        dehaze_net = nn.DataParallel(dehaze_net)
    train(args, dehaze_net)
    valid(args, dehaze_net)
    print('Time consume: %.2f h' % ((time.time()-t1) / 3600))
