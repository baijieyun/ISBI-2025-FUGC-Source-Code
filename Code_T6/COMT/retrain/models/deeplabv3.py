#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   deeplabv3.py
@Time    :   2025/02/10 17:26:11
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   None
'''

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")

class Deeplabv3(nn.Module):
    def __init__(self, type='deeplabv3_resnet50', pretrained=True):
        super(Deeplabv3, self).__init__()
        self.model = get_deeplabv3(type=type, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)['out']

def get_deeplabv3(type='deeplabv3_resnet50', pretrained=True):
    '''
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    '''
    if type == 'deeplabv3_mobilenet_v3_large':
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained)
    elif type == 'deeplabv3_resnet50':
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    elif type == 'deeplabv3_resnet101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
    else:
        raise ValueError("Invalid model type. Must be one of 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet50', 'deeplabv3_resnet101'.")
    # 修改模型输出通道数
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, 3, kernel_size=(1, 1), stride=(1, 1))
    return model
# # 1. 加载DeepLabV3模型
# model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
# in_channels = model.classifier[4].in_channels
# model.classifier[4] = nn.Conv2d(in_channels, 3, kernel_size=(1, 1), stride=(1, 1))
# x = torch.rand(3, 3, 336, 544)
# print(model(x)['out'].shape)

if __name__ == '__main__':
    model         = Deeplabv3(type='deeplabv3_resnet101')
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model         = model.to(device)
    import torch
    from thop import profile
    input1 = torch.randn(1, 3, 336, 544) 
    flops, params = profile(model, inputs=(input1, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


    # from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
    # import torch

    # x = torch.randn((1, 3, 336, 544)).cpu()
    # print('model parameters: ', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
    # flops = FlopCountAnalysis(model, x)
    # acts = ActivationCountAnalysis(model, x)
    # print(f"total flops : {(flops.total() + acts.total()) / 1e9}", 'G')
    # x = torch.rand((2,3,336,544))
    # out = model(x)['out']
    # print(out.shape)