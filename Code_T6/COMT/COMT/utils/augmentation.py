import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

class BasicTsf:
    """
    基础数据增强策略
    """
    def __init__(self, img_size = (336,544), long_mask=True):
        self.crop = (32, 32)
        self.p_flip = 0.5
        self.p_rota = 0.5
        self.p_gama = 0.2
        self.img_size = img_size
        color_jitter_params = (0.1, 0.1, 0.1, 0.1)
        self.color_tf = T.ColorJitter(*color_jitter_params)
        self.long_mask = long_mask
    def __call__(self, image, mask=None):
        if mask is None:
            image = image.transpose(1,2,0).astype(np.uint8)
            if np.random.rand() < self.p_gama:
                c = 1
                g = np.random.randint(10, 25) / 10.0
                # g = 2
                image = (np.power(image / 255, 1.0 / g) / c) * 255
                image = image.astype(np.uint8)
            # transforming to PIL image
            image = F.to_pil_image(image)
            # random crop
            if self.crop:
                i, j, h, w = T.RandomCrop.get_params(image, self.crop)
                # 将图像转换为numpy数组
                image_np = np.array(image)
                # 将裁剪区域置0
                image_np[i:i+h, j:j+w, :] = 0
                # 将numpy数组转回PIL图像
                image = F.to_pil_image(image_np)
            # random horizontal flip
            if np.random.rand() < self.p_flip:
                image = F.hflip(image)
            # random rotation
            if np.random.rand() < self.p_rota:
                angle = T.RandomRotation.get_params((-30, 30))
                image = F.rotate(image, angle)
            image_tea = image
            image = self.color_tf(image)
            image_tea = F.to_tensor(image_tea)
            image_tea = image_tea.numpy()
            image_tea = torch.as_tensor(image_tea, dtype=torch.float32)
            image = F.to_tensor(image)
            image = image.numpy()
            image = torch.as_tensor(image, dtype=torch.float32)
            return image, image_tea
        else:
            image = image.transpose(1,2,0).astype(np.uint8)
            mask = mask.transpose(1,2,0).astype(np.uint8)
            if np.random.rand() < self.p_gama:
                c = 1
                g = np.random.randint(10, 25) / 10.0
                # g = 2
                image = (np.power(image / 255, 1.0 / g) / c) * 255
                image = image.astype(np.uint8)
            # transforming to PIL image

            image, mask = F.to_pil_image(image), F.to_pil_image(mask)

            # random crop
            if self.crop:
                i, j, h, w = T.RandomCrop.get_params(image, self.crop)
                # 将图像转换为numpy数组
                image_np = np.array(image)
                mask_np = np.array(mask)
                
                # 将裁剪区域置0
                image_np[i:i+h, j:j+w, :] = 0
                mask_np[i:i+h, j:j+w] = 0
                
                # 将numpy数组转回PIL图像
                image = F.to_pil_image(image_np)
                mask = F.to_pil_image(mask_np)
            # random horizontal flip
            if np.random.rand() < self.p_flip:
                image, mask = F.hflip(image), F.hflip(mask)
            # random rotation
            if np.random.rand() < self.p_rota:
                angle = T.RandomRotation.get_params((-30, 30))
                image, mask = F.rotate(image, angle), F.rotate(mask, angle)
            # student strong augmentation
            image = self.color_tf(image)
            image = F.to_tensor(image)
            image = image.numpy()
            image = torch.as_tensor(image, dtype=torch.float32)

            if not self.long_mask:
                mask = F.to_tensor(mask)
            else:
                mask = to_long_tensor(mask)
            return image, mask
        