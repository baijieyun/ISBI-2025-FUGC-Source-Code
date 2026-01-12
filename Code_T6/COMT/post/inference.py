#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   inference.py
@Time    :   2025/01/20 01:50:40
@Author  :   biabuluo
@Version :   1.0
@Desc    :   模型推理 todo
"""

import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

# def apply_clahe(image_array):
#     if len(image_array.shape) == 3:
#         image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
#     image_array = np.uint8(image_array)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     clahe_image = clahe.apply(image_array)
#     clahe_image = np.expand_dims(clahe_image, axis=-1)  # 在最后一维增加一个维度
#     return np.repeat(clahe_image, 3, axis=-1)


class Inference:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.to(device)
        self.file_name = None

    def preprocess(self, image_path):
        """Preprocess input image."""
        self.file_name = str(image_path).split("/")[-1]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)  # Shape: [H, W, 3]
        # image = apply_clahe(image)
        image = np.transpose(image, (2, 0, 1))  # Reshape to [3, H, W]
        image = image / 255.0  # Normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float).unsqueeze(
            0
        )  # Shape: [1, 3, H, W]
        return image.to(self.device)

    def postprocess(self, prediction):
        """Postprocess model output."""
        pred = prediction.squeeze(0).argmax(dim=0).cpu().numpy()  # Shape: [H, W]
        return pred

    def savefig(self, pred, path):
        # 将预测结果转换为8位无符号整数
        pred_uint8 = pred.astype(np.uint8)
        save_path = os.path.join(path, self.file_name)
        # 创建图像并保存
        img = Image.fromarray(pred_uint8, mode="L")
        img.save(save_path, format="PNG")

    def predict(self, img):
        with torch.no_grad():
            prediction = self.model(img)
        return prediction

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


if __name__ == "__main__":
    from unet import U_Net

    model = U_Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(
            "/home/server/cy_home/challenge/baseline/outputs/experiment4/fold1/best_weight_unet.pth",
            map_location=device,
        )
    )
    inference = Inference(model, device)
    image_path = "/home/server/cy_home/challenge/data/labeled_data/images/0001.png"

    img = inference.preprocess(image_path)
    pred = inference.predict(img)
    pred_mask = inference.postprocess(pred)
    inference.savefig(pred_mask, "./")
