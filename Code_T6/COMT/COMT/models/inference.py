#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2025/01/20 01:50:40
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   模型推理 
'''

import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from models.unet import U_Net  # 替换为实际模型类

class Inference:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.to(device)
    def preprocess(self, image_path):
        """Preprocess input image."""
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)  # Shape: [H, W, 3]
        image = np.transpose(image, (2, 0, 1))  # Reshape to [3, H, W]
        image = image / 255.0  # Normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0)  # Shape: [1, 3, H, W]
        return image.to(self.device)

    def postprocess(self, prediction):
        """Postprocess model output."""
        pred = prediction.squeeze(0).argmax(dim=0).cpu().numpy()  # Shape: [H, W]
        return pred

    def infer(self, image_path, gt_path):
        """Run inference on a single image."""
        self.model.eval()
        image = self.preprocess(image_path)
        with torch.no_grad():
            prediction = self.model(image)
        pred_mask = self.postprocess(prediction)

        gt_mask = Image.open(gt_path).convert("L")
        gt_mask = np.array(gt_mask)

        return pred_mask, gt_mask

    def visualize(self, image_path, gt_mask, pred_mask, save_path):
        """Visualize input image, ground truth, and prediction."""
        image = Image.open(image_path).convert("L")
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="viridis")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="viridis")
        plt.title("Predicted Mask")
        plt.axis("off")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


