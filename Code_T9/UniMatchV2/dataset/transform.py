import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn as nn


def preprocess_mask(mask):
    """ Convert fractional values in the mask to integer class indices. """
    mask = np.array(mask)  # Convert to NumPy array if not already

    # Define the mapping (assuming 0.0 is background, 0.00392 is class 1, 0.00784 is class 2)
    mapping = {0.0: 0, 0.003921569: 1, 0.007843138: 2}

    # Use vectorized operations for speed
    mask_int = np.zeros_like(mask, dtype=np.uint8)  # Initialize with zeros
    for frac_value, class_index in mapping.items():
        mask_int[mask == frac_value] = class_index  # Map fractional values to integer labels

    return torch.tensor(mask_int, dtype=torch.long)  # Convert to tensor

def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if torch.rand(1).item() < p:  
        img = F.hflip(img)  
        mask = F.hflip(mask)  
    return img, mask


def normalize(img, mask=None):
    """Normalize image with mean & std but ensure mask remains unchanged."""
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)  # Convert only if not already tensor
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    
    if mask is not None and not isinstance(mask, torch.Tensor):
        mask = torch.tensor(np.array(mask), dtype=torch.long)  # Ensure correct dtype
    
    return (img, mask) if mask is not None else img

def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    """Apply Gaussian blur augmentation to a PyTorch tensor."""
    if random.random() < p:
        gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))  # Adjust kernel size and sigma
        img = gaussian_blur(img)
    return img


def pad_tensor(img, target_size):
    """Zero pad a tensor to match the target size (C, H, W) only if necessary."""
    _, h, w = img.shape
    pad_h = max(0, target_size[0] - h)  # Padding for height
    pad_w = max(0, target_size[1] - w)  # Padding for width

    if pad_h == 0 and pad_w == 0:
        return img  # No padding needed

    pad = nn.ZeroPad2d((0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    return pad(img)

def obtain_cutmix_box(img_h, img_w, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    """
    Generate a CutMix mask that randomly selects a region in the image for mixing.

    Args:
        img_h (int): Image height.
        img_w (int): Image width.
        p (float): Probability of applying CutMix (default: 0.5).
        size_min (float): Minimum area of the CutMix region relative to the image.
        size_max (float): Maximum area of the CutMix region relative to the image.
        ratio_1 (float): Minimum aspect ratio.
        ratio_2 (float): Maximum aspect ratio.

    Returns:
        torch.Tensor: A binary mask (1 for CutMix region, 0 elsewhere).
    """
    mask = torch.zeros((img_h, img_w), dtype=torch.uint8)  # Ensure correct dtype

    if random.random() > p:
        return mask  # Return an empty mask (no CutMix applied)

    # Ensure the CutMix box size does not exceed the image size
    max_area = img_h * img_w * size_max
    min_area = img_h * img_w * size_min

    for _ in range(10):  # Try multiple times to find a valid box
        area = np.random.uniform(min_area, max_area)
        aspect_ratio = np.random.uniform(ratio_1, ratio_2)

        cutmix_w = min(max(1, int(np.sqrt(area / aspect_ratio))), img_w)  # Ensure at least 1 pixel
        cutmix_h = min(max(1, int(np.sqrt(area * aspect_ratio))), img_h)  # Ensure at least 1 pixel

        x = np.random.randint(0, max(1, img_w - cutmix_w + 1))  # Ensure valid range
        y = np.random.randint(0, max(1, img_h - cutmix_h + 1))  # Ensure valid range

        if x + cutmix_w <= img_w and y + cutmix_h <= img_h:
            mask[y:y + cutmix_h, x:x + cutmix_w] = 1
            return mask  # Return a valid CutMix mask

    return mask  # If no valid CutMix region was found, return an empty mask

