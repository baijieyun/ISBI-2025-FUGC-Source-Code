#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   EBA.py
@Time    :   2025/02/11 17:52:50
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   Entropy-Based Selection
'''
import torch
import torch.nn.functional as F

def weighted_mse_loss(output, target, mu=1.0, b=0.5, is_wights=True):
    """
    计算加权MSE损失，熵高的像素点低损失，熵低的像素点高损失。
    
    参数:
    - output: 模型的输出，形状为 [batch_size, num_classes, height, width]
    - target: 伪标签，形状为 [batch_size, num_classes, height, width]
    - entropy_map: 熵图，形状为 [batch_size, height, width]
    - alpha: 控制权重反比关系的系数，默认为1，较大值使熵低像素的权重增加
    
    返回:
    - loss: 加权后的MSE损失
    """
    # 计算MSE损失
    mse_loss = (output-target)**2
    # target_soft = torch.softmax(target, dim=1)
    entropy_map = calculate_entropy(target)
    # 根据熵调整权重，熵高的像素点低损失，熵低的像素点高损失
    # 使用熵的反比作为权重，可以通过公式：w = exp(-alpha * entropy_map)
    weights = torch.exp(-mu * entropy_map.unsqueeze(1)) + b  # 在类别维度上扩展
    if is_wights:
        mse_loss = mse_loss * weights
    # 对每个像素求平均损失
    loss = mse_loss.mean()
    return loss

def ebs(x1_soft, x2_soft):
    x1_entropy = calculate_entropy(x1_soft)
    x2_entropy = calculate_entropy(x2_soft)
    pesudo_output = select_output_by_entropy(x1_soft, x2_soft, x1_entropy, x2_entropy)
    return pesudo_output


def calculate_entropy(softmax_output, dim=1):
    """
    计算经过softmax激活后的张量在指定维度上的熵图。
    
    参数:
    - softmax_output: 经过softmax的张量，形状为 [batch_size, num_classes, height, width]
    - dim: 对softmax输出在该维度上计算熵，默认为 1 (即按类别维度计算)

    返回:
    - entropy_map: 计算得到的熵图，形状为 [batch_size, height, width]
    """
    entropy = -torch.sum(softmax_output * torch.log(softmax_output + 1e-10), dim=dim)
    return entropy

def select_output_by_entropy(output1, output2, entropy_map1, entropy_map2):
    """
    根据两个熵图选择模型输出，每个像素点选择熵最小的对应模型输出。
    
    参数:
    - output1: 第一个模型的输出，形状为 [batch_size, num_classes, height, width]
    - output2: 第二个模型的输出，形状为 [batch_size, num_classes, height, width]
    - entropy_map1: 第一个熵图，形状为 [batch_size, height, width]
    - entropy_map2: 第二个熵图，形状为 [batch_size, height, width]
    
    返回:
    - selected_output: 选择熵最小的模型输出，形状为 [batch_size, num_classes, height, width]
    """
    # 对每个像素点，选择熵较小的模型的输出
    selected_output = torch.where(entropy_map1.unsqueeze(1) < entropy_map2.unsqueeze(1),output1, output2)
    return selected_output
