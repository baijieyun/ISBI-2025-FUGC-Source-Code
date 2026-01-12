import os
import random
from utils.dataloader import *
from utils.augmentation import *
import random
import re
from torch import nn
import time
import torch
import h5py
from scipy.ndimage import label, sum as ndi_sum
from model.semseg.dpt import DPT
import SimpleITK as sitk
from torchvision.utils import save_image
# from model.unet import UNet
from PIL import Image
from skimage.feature import local_binary_pattern


class DSC:
    def __init__(self):
        self.smooth = 1e-6

    def __call__(self, y_pred, y_truth):
        """
        :param y_pred: (H, W)
        :param y_truth: (H, W)
        :return:
        """
        y_pred_f = np.eye(3)[y_pred.reshape(-1)]  # (H*W, 3)
        y_truth_f = np.eye(3)[y_truth.reshape(-1)]  # (H*W, 3)

        dice1 = (2. * np.sum(y_pred_f[:, 1] * y_truth_f[:, 1]) + self.smooth) / (
                np.sum(y_pred_f[:, 1]) + np.sum(y_truth_f[:, 1]) + self.smooth)
        dice2 = (2. * np.sum(y_pred_f[:, 2] * y_truth_f[:, 2]) + self.smooth) / (
                np.sum(y_pred_f[:, 2]) + np.sum(y_truth_f[:, 2]) + self.smooth)

        return dice1, dice2


class HD(nn.Module):
    def __init__(self):
        super(HD,self).__init__()
        pass

    def numpy_to_image(self, image) -> sitk.Image:
        image = sitk.GetImageFromArray(image)
        return image

    def evaluation(self, pred: sitk.Image, label: sitk.Image):
        result = dict()

        # 计算upper指标
        pred_data_upper = sitk.GetArrayFromImage(pred)
        pred_data_upper[pred_data_upper == 2] = 0
        pred_upper = sitk.GetImageFromArray(pred_data_upper)

        label_data_upper = sitk.GetArrayFromImage(label)
        label_data_upper[label_data_upper == 2] = 0
        label_upper = sitk.GetImageFromArray(label_data_upper)


        result['hd_upper'] = float(self.cal_hd(pred_upper, label_upper))

        # 计算lower指标
        pred_data_lower = sitk.GetArrayFromImage(pred)
        pred_data_lower[pred_data_lower == 1] = 0
        pred_data_lower[pred_data_lower == 2] = 1
        pred_lower = sitk.GetImageFromArray(pred_data_lower)

        label_data_lower = sitk.GetArrayFromImage(label)
        label_data_lower[label_data_lower == 1] = 0
        label_data_lower[label_data_lower == 2] = 1
        label_lower = sitk.GetImageFromArray(label_data_lower)


        result['hd_lower'] = float(self.cal_hd(pred_lower, label_lower))
        
        # 计算总体指标
        pred_data_all = sitk.GetArrayFromImage(pred)
        pred_data_all[pred_data_all == 2] = 1
        pred_all = sitk.GetImageFromArray(pred_data_all)

        label_data_all = sitk.GetArrayFromImage(label)
        label_data_all[label_data_all == 2] = 1
        label_all = sitk.GetImageFromArray(label_data_all)
        
        result['hd_all'] = float(self.cal_hd(pred_all, label_all))
        
        return (result['hd_all'] + result['hd_lower'] + result['hd_upper']) / 3

    def forward(self, pred, label):
        """
        :param pred: (BS,3,336,544)
        :param label: (BS,336,544)
        :return:
        """
        # print(pred.shape)
        # print(label.shape)

        pred = pred.astype(np.int64)  # (H,W) value:0,1,2  1-upper 2-lower
        label = label.astype(np.int64) # (H,W) value:0,1,2  1-upper 2-lower
        pre_image = self.numpy_to_image(pred)
        truth_image = self.numpy_to_image(label)

        result = self.evaluation(pre_image, truth_image)

        return result
    def cal_hd(self, a, b):
        a = sitk.Cast(sitk.RescaleIntensity(a), sitk.sitkUInt8)
        b = sitk.Cast(sitk.RescaleIntensity(b), sitk.sitkUInt8)
        if np.count_nonzero(a) == 0 or np.count_nonzero(b) == 0:
            
            hd=float('inf')
            
        else:
            filter1 = sitk.HausdorffDistanceImageFilter()
            filter1.Execute(a, b)
            hd = filter1.GetHausdorffDistance()
        return hd



hd=HD()
dsc=DSC()
def get_model():
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs['base'], 'nclass': 3})
    
    model_path='./eval_model/dpt_small_freeze.pth'
    # model_path='./eval_model/dpt_small_nofre.pth'
    # model_path='./eval_model/dpt_small_fre6to11.pth'
    # model_path='./eval_model/dpt_base_freeze.pth'
    # model_path='./eval_model/dpt_base_nofre.pth'

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_ema'])
    return model


def postprocess(input_array):
    result = np.zeros_like(input_array)
    
    for class_value in [0, 1, 2]:  # 遍历三种像素值
        mask = (input_array == class_value)  # 获取当前类别的掩码
        labeled_array, num_features = label(mask)  # 连通域标记
        
        if num_features > 0:
            # 计算每个连通域的大小
            sizes = ndi_sum(mask, labeled_array, index=range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1  # 找到最大连通域的标签（索引从1开始）
            result[labeled_array == largest_label] = class_value  # 仅保留最大连通域
    
    return result

def get_largest_connected_component(labeled_array, num_features):
    largest_cc = None
    max_size = 0

    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        size = np.sum(component)
        if size > max_size:
            max_size = size
            largest_cc = component

    return largest_cc

def calculate_lbp_similarity(image1):
    radius = 3
    n_points = 8 * radius
    lbp1 = local_binary_pattern(image1, n_points, radius, method="uniform")
    hist1, _ = np.histogram(lbp1, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist1 = hist1 / np.sum(hist1)
    return hist1

def predict(model, X):
    """
    X: numpy array of shape (3,336,544)
    """


    model.eval()
    X = X / 255.0
    X = X.transpose(2,0,1)
    X = np.pad(X, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=0)

    # sim = np.argmax(np.sum(np.sqrt(supporthist * histx),axis=1))


    image = torch.tensor(X, dtype=torch.float).unsqueeze(0).cuda()

    seg = model(image)  # seg (1,3,336,544)

    seg = seg.squeeze(0).argmax(dim=0).detach().cpu().numpy()  # (336,544) values:{0,1,2} 1 upper 2 lower

    seg = seg[:, 1:-1]
    seg = postprocess(seg)
    return seg

def read_images_and_labels_from_h5(test_path):
    image_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.h5')]
    images = []
    labels = []
    for h5_file in image_files:
        with h5py.File(h5_file, 'r') as f:
            img = f['image'][:]
            label = f['label'][:]
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)



def main():
    # ========================== set random seed ===========================#
    seed_value = 2024  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    # ========================== set hyper parameters =========================#


    # ========================== get model, dataloader, optimizer and so on =========================#

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # ========================= test =======================#
    test_path='./oursdataPath/val_data'
    images,labels = read_images_and_labels_from_h5(test_path)
    model = get_model()
    model = model.to(device)


    runTime=0
    
    dsc_all=[]
    hd_all=[]
    for _, (img,label) in enumerate(zip(images,labels)):
        start = time.perf_counter()
        seg = predict(model,img)
        end = time.perf_counter()
        runTime=runTime+end-start
        seg=np.expand_dims(seg,0)
        label=np.expand_dims(label,0)
        dsc1,dsc2=dsc(seg,label)
        dsc_one=(dsc1+dsc2)/2
        hd_one=hd(seg,label)
        dsc_all.append(dsc_one)
        hd_all.append(hd_one)
        print('DSC:',dsc_one,'HD:',hd_one,'Running Time:',end-start)
    dsc_mean=np.mean(dsc_all)
    hd_mean=np.mean(hd_all)
    print('DSC Mean:',dsc_mean,'HD Mean:',hd_mean,'Running Time Mean:',runTime)

        






if __name__ == '__main__':
    main()