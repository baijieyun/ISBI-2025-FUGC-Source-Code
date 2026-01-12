#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   ulite.py
@Time    :   2025/02/06 16:23:39
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   None
'''
from thop import clever_format, profile
import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding=(max(h // 2, dilation), 0), groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding=(0, max(w // 2, dilation)), groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel = (7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x
    
class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""
    def __init__(self, dim):
        super().__init__()

        gc = dim//4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw2 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.dw3 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 3)

        self.bn = nn.BatchNorm2d(4*gc)
        self.pw2 = nn.Conv2d(4*gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x

class ULite(nn.Module):
    def __init__(self):
        super().__init__()
        n1 = 8
        filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1*32]
        """Encoder"""
        self.conv_in = nn.Conv2d(3, filter[0], kernel_size=7, padding=3)
        self.e1 = EncoderBlock(filter[0], filter[1])
        self.e2 = EncoderBlock(filter[1], filter[2])
        self.e3 = EncoderBlock(filter[2], filter[3])
        self.e4 = EncoderBlock(filter[3], filter[4])
        self.e5 = EncoderBlock(filter[4], filter[5])

        """Bottle Neck"""
        self.b5 = BottleNeckBlock(filter[5])

        """Decoder"""
        self.d5 = DecoderBlock(filter[5], filter[4])
        self.d4 = DecoderBlock(filter[4], filter[3])
        self.d3 = DecoderBlock(filter[3], filter[2])
        self.d2 = DecoderBlock(filter[2], filter[1])
        self.d1 = DecoderBlock(filter[1], filter[0])
        self.conv_out = nn.Conv2d(filter[0], 3, kernel_size=1)

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)         # (512, 8, 8)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x

if __name__ == '__main__':
    model         = ULite()
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model         = model.to(device)
   

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
    import torch

    x = torch.randn((1, 3, 336, 544)).cpu()
    print('model parameters: ', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
    flops = FlopCountAnalysis(model, x)
    acts = ActivationCountAnalysis(model, x)
    print(f"total flops : {(flops.total() + acts.total()) / 1e9}", 'G')
    x = torch.rand((2,3,336,544))
    out = model(x)
    print(out.shape)
