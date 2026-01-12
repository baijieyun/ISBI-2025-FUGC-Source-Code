from dataset.transform import *

import math  
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        
        # print(f"Opening file: {id_path}")
        with open(id_path, 'r') as f:
            # print(f"Opening file: {id_path}")
            self.ids = f.read().splitlines()
        
        # Handle oversampling for labeled data
        if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]

    def __getitem__(self, item):
        id_split = self.ids[item].split(" ")  # Splitting image & mask paths if available
        img_path = os.path.join(self.root, id_split[0])
        img = Image.open(img_path).convert('RGB')
        
        img = transforms.ToTensor()(img) 

        # Handling labeled and unlabeled cases properly
        if self.mode == 'train_u' or len(id_split) == 1:  # Unlabeled images → Use dummy masks
            mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.uint8)  # (1, H, W)
        else:  # Labeled images → Read the actual mask
            mask_path = os.path.join(self.root, id_split[1])
            mask = Image.open(mask_path)
            mask = torch.tensor(np.array(mask), dtype=torch.long).unsqueeze(0)  # Convert to (1, H, W)
            
        if self.mode == 'val':  # Validation set only normalizes images and masks
            img, mask = normalize(img, mask)
            return img, mask, id_split[0]
        
        target_size = (336, 560)  # (Height, Width)
        img = pad_tensor(img, target_size)
        mask = pad_tensor(mask.float(), target_size).long()  # Ensure integer mask

        # Apply horizontal flip augmentation
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)
        
        # Create weak and strong augmented versions
        img_w, img_s1, img_s2 = img.clone(), img.clone(), img.clone()

        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)

        cutmix_box1 = obtain_cutmix_box(img_s1.shape[1], img_s1.shape[2], p=0.5)  # height, width
        
        # if random.random() < 0.8:
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.shape[1], img_s2.shape[2], p=0.5)  # height, width

        ignore_mask = torch.zeros_like(mask)  # (1, H, W), Ensure uint8


        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
