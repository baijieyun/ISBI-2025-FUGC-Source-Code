#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   aux.py
@Time    :   2025/02/18 00:50:36
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   None
'''
from models.deeplabv3 import get_deeplabv3, Deeplabv3
from models.unet import U_Net
from models.factory.tiny_unet import TinyUNet
from models.factory.ulite import ULite
from models.factory.unext import UNext

def get_aux(type='deeplabv3_resnet50'):
    if type == 'unet16':
        model = U_Net(n1=16)
    elif type == 'unet32':
        model = U_Net(n1=32)
    elif type == 'unet64':
        model = U_Net(n1=64)
    elif type == 'tinyunet':
        model = TinyUNet(in_channels=3, num_classes=3)
    elif type == 'ulite':
        model = ULite()
    elif type == 'unext':
        model = UNext(num_classes=3)
    else:
        model = Deeplabv3(type=type)
    return model