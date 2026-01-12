from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box,random_rot_flip2,random_rotate2

from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2 as cv


class MyDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('oursdataPath/val.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self.root, id), 'r')
        img = sample['image'][:]
        # img_2=np.stack((img_1, img_1, img_1), axis=-1)
        img = np.pad(img, ((104, 104), (0, 0), (0, 0)), mode='constant', constant_values=0)
        # img = np.pad(img, ((91, 91), (0, 0), (0, 0)), mode='constant', constant_values=0)

        if self.mode != 'train_u':
            mask = sample['label'][:]
            # mask = np.pad(mask, ((91, 91), (0, 0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((104, 104), (0, 0)), mode='constant', constant_values=0)

        # print(f"Shape of img: {img.shape}")
        # print(f"Shape of mask: {mask.shape}")

        if self.mode == 'val':
            return torch.from_numpy(img).float()/255.0, torch.from_numpy(mask).long()


        # img = cv.resize(img, (self.size, self.size))
        # mask = cv.resize(mask, (self.size, self.size))

        

        if random.random() > 0.5:
            if self.mode != 'train_u':
                img, mask = random_rot_flip(img, mask)
            else:
                img = random_rot_flip2(img)
        elif random.random() > 0.5:
            if self.mode != 'train_u':
                img, mask = random_rotate(img, mask)
            else:
                img = random_rotate2(img)
        x, y, z = img.shape
        img = zoom(img, (self.size / x, self.size / y, 1), order=0) 
        if self.mode != 'train_u':
            mask = zoom(mask, (self.size / x, self.size / y), order=0) 
        



        # print(f"Shape of img: {img.shape}")
        # print(f"Shape of mask: {mask.shape}")

        if self.mode == 'train_l':
            return torch.from_numpy(img).permute(2, 0, 1).float()/255.0, torch.from_numpy(mask).long()

        img = Image.fromarray(img.astype(np.uint8))
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).permute(2, 0, 1).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).permute(2, 0, 1).float() / 255.0

        return img, img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
