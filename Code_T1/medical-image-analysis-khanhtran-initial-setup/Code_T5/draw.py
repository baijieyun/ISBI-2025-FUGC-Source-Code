import os
import random
from utils.dataloader import *
from utils.augmentation import *
import random
import re
from model.unet import UNet
from model.attention_unet import AttU_Net
from scipy.ndimage import label, sum as ndi_sum
from model.semseg.dpt import DPT
from utils.criterion import MyCriterion
from torch.utils.data import DataLoader
from utils.metrics import DSC, HD
from torchvision.utils import save_image
# from model.unet import UNet
from PIL import Image
from skimage.feature import local_binary_pattern

def save_segmentation_with_contours(img,seg, path):
    # 将原始图像转换为 RGB 格式
    # img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 找到类别 1 和类别 2 的轮廓并绘制
    for class_value, contour_color in zip([1, 2], [(255, 0, 0), (0, 255, 0)]):  # 红色和绿色
        mask = (seg == class_value).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, contour_color, 2)  # 用指定颜色勾勒轮廓

    seg_img = Image.fromarray(img)
    seg_img.save(path)
image_file='./oursdataPath/val_data_png/images/0002.png'
label_file='./oursdataPath/val_data_png/labels/0030.png'
img = Image.open(image_file)
img = np.array(img)
label=Image.open(label_file)
label=np.array(label)
save_segmentation_with_contours(img,label, f'./oursdataPath/val_data_png/0002.png')
