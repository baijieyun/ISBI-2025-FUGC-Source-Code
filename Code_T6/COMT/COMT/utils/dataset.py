#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2025/02/08 19:30:26
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   dataset
'''
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import numpy as np
import torch
import os.path
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class LabeledDataSets(Dataset):
    """ 适用于 Mean Teacher 的数据集 """
    def __init__(self, dir, ls, tsf=None):
        self.dir = dir
        self.tsf = tsf
        images = []
        labels = []

        for file in ls:
            image_path = os.path.join(dir, "images", file)
            label_path = os.path.join(dir, "labels", file)
            image = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(image).transpose(2,0,1)
            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label)
            images.append(np.array([image]))
            labels.append(np.array([label]))

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        self.images = images
        self.labels = labels
        print("**有标记数据**")
        print(f"num_imgs:{len(images)}\tnum_labels:{len(labels)}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = correct_dims(self.images[idx])
        label = np.array([self.labels[idx]])
        # image, image_stu, mask
        image , label = self.tsf(image, label)
        return image, label



class UnlabeledDataSets(Dataset):
    def __init__(
        self,
        dir,
        ls,
        tsf,
    ):
        self.tsf = tsf
        unlabeled_path = os.path.join(dir, "images")
        ls = os.listdir(unlabeled_path)
        images = []
        for file in ls:
            image_path = os.path.join(unlabeled_path, file)
            image = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(image).transpose(2,0,1)
            images.append(np.array([image]))

        images = np.concatenate(images, axis=0)
        self.images = images
        self.labels = None
        print("**无标记数据**")
        print(f"num_imgs:{len(images)}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = correct_dims(self.images[idx])
    
        image, image_w = self.tsf(image)
        return image, image_w

