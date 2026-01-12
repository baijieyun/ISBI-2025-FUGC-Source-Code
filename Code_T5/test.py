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

def get_model():
    # model_configs = {
    #     'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #     'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    # }
    # model = DPT(**{**model_configs['base'], 'nclass': 3})
    # model_path='./eval_model/dpt_small_freeze.pth'
    # model_path='./eval_model/dpt_small_nofre.pth'
    # model_path='./eval_model/dpt_base_freeze.pth'
    model = UNet(3,3)
    model_path='./eval_model/UNet.pth'
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

def get_largest_connected_component(labeled_array, num_features):
    largest_cc = None
    max_size = 0
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
    # X = np.pad(X, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=0)

    # sim = np.argmax(np.sum(np.sqrt(supporthist * histx),axis=1))


    image = torch.tensor(X, dtype=torch.float).unsqueeze(0).cuda()

    seg = model(image)  # seg (1,3,336,544)

    seg = seg.squeeze(0).argmax(dim=0).detach().cpu().numpy()  # (336,544) values:{0,1,2} 1 upper 2 lower

    # seg = seg[:, 1:-1]
    seg = postprocess(seg)
    return seg
# def predict(model, X):
#     """
#     X: numpy array of shape (3,336,544)
#     """
#     model.eval()
#     X = X / 255.0
#     X = np.pad(X, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=0)
#     print(X.shape)
#     image = torch.tensor(X, dtype=torch.float).unsqueeze(0).cuda()

#     seg = model(image)  # seg (1,3,336,544)

#     seg = seg.squeeze(0).argmax(dim=0).detach().cpu().numpy()  # (336,544) values:{0,1,2} 1 upper 2 lower

#     seg = seg[:, 1:-1]
#     return seg

def read_images_to_numpy(test_path):
    image_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.png')]
    images = []
    for image_file in image_files:
        img = Image.open(image_file)
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        images.append(img)
    return np.array(images)

def save_segmentation(seg, path):

    color_map = {
        0: [0, 0, 0],       # 黑色
        1: [255, 0, 0],     # 红色
        2: [0, 255, 0]      # 绿色
    }
    seg_rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        seg_rgb[seg == label] = color
    seg_img = Image.fromarray(seg_rgb)
    seg_img.save(path)


def save_segmentation_with_contours(img,seg, path):
    # 将原始图像转换为 RGB 格式
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 找到类别 1 和类别 2 的轮廓并绘制
    for class_value, contour_color in zip([1, 2], [(255, 0, 0), (0, 255, 0)]):  # 红色和绿色
        mask = (seg == class_value).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, contour_color, 2)  # 用指定颜色勾勒轮廓

    seg_img = Image.fromarray(img)
    seg_img.save(path)


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
    test_path='./oursdataPath/val_data_png/images'
    images = read_images_to_numpy(test_path)
    model = get_model()
    model = model.to(device)




    for idx, img in enumerate(images):
        seg = predict(model,img)
        print(seg.shape)
        
        seg_img = Image.fromarray(seg.astype(np.uint8))
        # seg_img.save(f'./oursdataPath/val_data_png/preds/{idx + 1:04d}.png')
        # save_segmentation(seg, f'./oursdataPath/unlabeled_data_png/labels/{idx + 1:04d}.png')
        save_segmentation_with_contours(img,seg, f'./oursdataPath/val_data_png/{idx + 1:04d}.png')






if __name__ == '__main__':
    main()