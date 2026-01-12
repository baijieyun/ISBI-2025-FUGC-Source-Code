#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   visualization.py
@Time    :   2025/01/20 01:43:01
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   绘制训练过程指标变化+推理结果可视化
'''

import matplotlib.pyplot as plt
import os

def plot_training_metrics(metrics, save_path):
    """
    Plot training metrics including loss and validation metrics.

    Args:
        metrics (list): A list of dictionaries with metrics for each epoch.
        save_path (str): Path to save the plot.
    """
    epochs = [int(m["epoch"]) for m in metrics]
    train_loss = [float(m["train_loss"]) for m in metrics]
    hd = [float(m["hd"]) for m in metrics]
    hd_upper = [float(m["hd_upper"]) for m in metrics]
    hd_lower = [float(m["hd_lower"]) for m in metrics]
    hd_all = [float(m["hd_all"]) for m in metrics]
    dsc = [float(m["dsc"]) for m in metrics]
    dsc_1 = [float(m["dsc_1"]) for m in metrics]
    dsc_2 = [float(m["dsc_2"]) for m in metrics]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, dsc, label="DSC", marker="o")
    plt.plot(epochs, dsc_1, label="DSC1", marker="o")
    plt.plot(epochs, dsc_2, label="DSC2", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("DSC")
    plt.title("Validation DSC Curve")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, hd, label="HD", marker="o")
    plt.plot(epochs, hd_upper, label="HD_upper", marker="o")
    plt.plot(epochs, hd_lower, label="HD_lower", marker="o")
    plt.plot(epochs, hd_all, label="HD_all", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("HD")
    plt.title("Validation HD Curve")
    plt.legend()


    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

import numpy as np

def visualize_segmentation(predictions, ground_truths, images, save_dir):
    """
    Visualize segmentation results with the input image, prediction, and ground truth.

    Args:
        predictions (list of np.array): List of predicted masks.
        ground_truths (list of np.array): List of ground truth masks.
        images (list of np.array): List of original images.
        save_dir (str): Directory to save visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, (pred, gt, img) in enumerate(zip(predictions, ground_truths, images)):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        save_path = os.path.join(save_dir, f"visualization_{i}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

