#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   load_model.py
@Time    :   2025/01/20 07:36:00
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   None
'''
from models.unet import U_Net
import torch

def load_model(model_path, device):
    model = U_Net()  
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
    