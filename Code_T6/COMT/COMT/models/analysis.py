#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   analysis.py
@Time    :   2025/01/20 01:58:15
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   None
'''

from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
import torch
from models.unet import U_Net
import time
from tqdm import tqdm

def analysis():
    x = torch.randn((1, 3, 544, 336)).cpu()

    model = U_Net().cpu()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print('model parameters: ', params, 'M')
    flops = FlopCountAnalysis(model, x)
    acts = ActivationCountAnalysis(model, x)
    flops = (flops.total() + acts.total()) / 1e9
    print(f"total flops : {flops}", 'G')
    x = torch.rand((2,3,544,336))
    out = model(x)
    print(f"Output shape: {out.shape}")

    # 测试模型推理速度
    num_iterations = 10
    total_time = 0
    
    # 预热
    for _ in range(10):
        _ = model(x)
    
    # 计时推理
    progress_bar = tqdm(range(num_iterations), desc="Inference Progress")
    for _ in progress_bar:
        start_time = time.time()
        _ = model(x)
        end_time = time.time()
        iteration_time = end_time - start_time
        total_time += (end_time - start_time)

        # 更新进度条描述
        progress_bar.set_postfix({"Iteration Time": f"{iteration_time:.4f}s"})
    
    avg_inference_time = total_time / num_iterations
    inference_speed = 1 / avg_inference_time
    
    print(f"Average inference time: {avg_inference_time:.4f} seconds")
    print(f"Inference speed: {inference_speed:.2f} FPS")

    return params, flops, avg_inference_time, inference_speed

if __name__ == '__main__':
    analysis()
