#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   predict.py
@Time    :   2025/02/21 19:38:19
@Author  :   biabuluo
@Version :   1.0
@Desc    :   None
"""
import torch
import os
from inference import Inference
from tqdm import tqdm
import numpy as np
import scipy.ndimage
from unet import U_Net  # 替换为实际模型类


def load_model(model_path, device):
    model = U_Net(n1=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


import numpy as np
import scipy.ndimage

import numpy as np
import scipy.ndimage
import numpy as np
import torch
from skimage.morphology import closing, square, remove_small_holes
from scipy.ndimage import label
import os


def post(seg):
    # 后处理：闭运算
    seg_processed = np.zeros_like(seg)

    for fg_class in [1, 2]:  # 对每个前景类别进行处理
        # 先进行闭运算
        seg_label = closing(seg == fg_class, square(7))  # 使用7*7的结构元素进行闭运算
        seg_label = remove_small_holes(seg_label, area_threshold=10000)
        # 最大联通域处理
        labeled_array, num_features = label(seg_label)  # 使用重新命名的label函数
        largest_component = 0
        largest_area = 0
        for region in range(1, num_features + 1):
            area = np.sum(labeled_array == region)
            if area > largest_area:
                largest_area = area
                largest_component = region
        # 保留最大联通域
        seg_processed[labeled_array == largest_component] = fg_class
    return seg_processed


def labeling(unlabeled_dir, model_path, device):
    # path list
    unlabeled_ls = [
        os.path.join(unlabeled_dir, "images", i)
        for i in os.listdir(os.path.join(unlabeled_dir, "images"))
    ]
    store_dir = "./pseudo_test2/"
    os.makedirs(store_dir, exist_ok=True)
    # 获取模型
    model = load_model(model_path, device)
    # 对未标记的样本进行标记
    inference = Inference(model, device)
    tbar = tqdm(unlabeled_ls)
    for image_path in tbar:
        img = inference.preprocess(image_path)
        pred = inference.predict(img)
        pred_mask = inference.postprocess(pred)
        pred_mask = post(pred_mask)
        inference.savefig(pred_mask, store_dir)
    print("labeling done!")


if __name__ == "__main__":
    model_path = "/home/chenyu/retrain/post/model.pth"
    unlabeled_dir = "/home/chenyu/retrain/data/unlabeled_data"
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    labeling(unlabeled_dir, model_path, device)
