import torch
import torch.nn as nn
import torch.nn.functional as F

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class EnhancedResBlock(nn.Module):
    def __init__(self, channel, dilation=1):
        super(EnhancedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=dilation, dilation=dilation, bias=True)
        self.in1 = nn.InstanceNorm2d(channel, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.in2 = nn.InstanceNorm2d(channel, affine=True)
        self.ca = CALayer(channel)
        self.pa = PALayer(channel)

    def forward(self, x):
        res = self.relu(self.in1(self.conv1(x)))
        res = self.in2(self.conv2(res))
        res = self.ca(res)
        res = self.pa(res)
        return self.relu(x + res)

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MultiScaleBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channel, out_channel // 2, 3, padding=1, bias=True)
        self.conv5x5 = nn.Conv2d(in_channel, out_channel // 2, 5, padding=2, bias=True)
    def forward(self, x):
        return torch.cat([self.conv3x3(x), self.conv5x5(x)], dim=1)

class dehaze_net(nn.Module):
    def __init__(self, gps=3, dim=64):
        super(dehaze_net, self).__init__()
        self.dim = dim
        
        # Initial Multi-Scale Feature Extraction
        self.init_feat = MultiScaleBlock(3, 32)
        self.conv1 = nn.Conv2d(32, self.dim, 3, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(self.dim, affine=True)
        self.relu = nn.ReLU(inplace=True)

        # Dilated Residual Blocks with Attention
        self.res1 = EnhancedResBlock(self.dim, dilation=1)
        self.res2 = EnhancedResBlock(self.dim, dilation=2)
        self.res3 = EnhancedResBlock(self.dim, dilation=4)
        self.res4 = EnhancedResBlock(self.dim, dilation=1)
        self.res5 = EnhancedResBlock(self.dim, dilation=2)
        self.res6 = EnhancedResBlock(self.dim, dilation=4)
        self.res7 = EnhancedResBlock(self.dim, dilation=1)

        # Feature Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(self.dim * 3, self.dim, 1, bias=True),
            CALayer(self.dim),
            PALayer(self.dim)
        )

        # Output Stage (Refinement)
        self.refine = nn.Sequential(
            nn.Conv2d(self.dim, 32, 3, padding=1, bias=True),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1, bias=True)
        )

    def forward(self, x):
        feat = self.relu(self.init_feat(x))
        y = self.relu(self.in1(self.conv1(feat)))

        y1 = self.res1(y)
        y = self.res2(y1)
        y = self.res3(y)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        # Global Skip Connection via Concatenation + Fusion
        concat = torch.cat([y1, y2, y3], dim=1)
        fused = self.fusion(concat)
        
        # Final prediction with global residual learning
        out = self.refine(fused)
        return self.relu(x + out)

if __name__ == '__main__':
    net = dehaze_net().cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    y = net(x)
    print("Output shape:", y.shape)
    num_params = sum(p.numel() for p in net.parameters())
    print("Total parameters:", num_params)

			

			
			






