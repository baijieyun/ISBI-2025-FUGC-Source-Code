from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
import torch
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

def weighted_mse_loss(output, target, entropy_map, alpha=1.0):
    """
    计算加权MSE损失，熵高的像素点低损失，熵低的像素点高损失。
    
    参数:
    - output: 模型的输出，形状为 [batch_size, num_classes, height, width]
    - target: 伪标签，形状为 [batch_size, num_classes, height, width]
    - entropy_map: 熵图，形状为 [batch_size, height, width]
    - alpha: 控制权重反比关系的系数，默认为1，较大值使熵低像素的权重增加
    
    返回:
    - loss: 加权后的MSE损失
    """
    # 计算MSE损失
    mse_loss = F.mse_loss(output, target, reduction='none')
    
    # 根据熵调整权重，熵高的像素点低损失，熵低的像素点高损失
    # 使用熵的反比作为权重，可以通过公式：w = exp(-alpha * entropy_map)
    weights = torch.exp(-alpha * entropy_map.unsqueeze(1))  # 在类别维度上扩展
    weighted_loss = mse_loss * weights
    
    # 对每个像素求平均损失
    loss = weighted_loss.mean()
    return loss

def calculate_entropy(softmax_output, dim=1):
    """
    计算经过softmax激活后的张量在指定维度上的熵图。
    
    参数:
    - softmax_output: 经过softmax的张量，形状为 [batch_size, num_classes, height, width]
    - dim: 对softmax输出在该维度上计算熵，默认为 1 (即按类别维度计算)

    返回:
    - entropy_map: 计算得到的熵图，形状为 [batch_size, height, width]
    """
    entropy = -torch.sum(softmax_output * torch.log(softmax_output + 1e-10), dim=dim)
    return entropy

def select_output_by_entropy(output1, output2, entropy_map1, entropy_map2):
    """
    根据两个熵图选择模型输出，每个像素点选择熵最小的对应模型输出。
    
    参数:
    - output1: 第一个模型的输出，形状为 [batch_size, num_classes, height, width]
    - output2: 第二个模型的输出，形状为 [batch_size, num_classes, height, width]
    - entropy_map1: 第一个熵图，形状为 [batch_size, height, width]
    - entropy_map2: 第二个熵图，形状为 [batch_size, height, width]
    
    返回:
    - selected_output: 选择熵最小的模型输出，形状为 [batch_size, num_classes, height, width]
    """
    # 对每个像素点，选择熵较小的模型的输出
    selected_output = torch.where(entropy_map1.unsqueeze(1) < entropy_map2.unsqueeze(1),output1, output2)
    return selected_output

# 示例用法
batch_size = 4
num_classes = 3
height = 256
width = 256

# 模拟两个模型的输出
output1 = torch.rand(batch_size, num_classes, height, width)
output2 = torch.rand(batch_size, num_classes, height, width)
output1 = F.softmax(output1, dim=1)  # 经过softmax激活
output2 = F.softmax(output2, dim=1)  # 经过softmax激活

# 计算两个熵图
entropy_map1 = calculate_entropy(output1, dim=1)
entropy_map2 = calculate_entropy(output2, dim=1)

# 根据熵最小选择输出
selected_output = select_output_by_entropy(output1, output2, entropy_map1, entropy_map2)

print(selected_output.shape)  # 输出选择后的张量形状，应该是 [batch_size, num_classes, height, width]



# x = torch.rand(3, 3, 336, 544)


# swin_unet = SwinTransformerSys(img_size=(336, 544),
#                         patch_size=(6, 8),
#                         in_chans=3,
#                         num_classes=3,
#                         embed_dim=96,
#                         depths=[ 2, 2, 2, 2 ],
#                         num_heads=[ 3, 6, 12, 24 ],
#                         window_size=2,
#                         mlp_ratio=4.,
#                         qkv_bias=True,
#                         qk_scale=None,
#                         drop_rate=0.0,
#                         drop_path_rate=0.1,
#                         ape=False,
#                         patch_norm=True,
#                         use_checkpoint=False)

# print(swin_unet(x).shape)