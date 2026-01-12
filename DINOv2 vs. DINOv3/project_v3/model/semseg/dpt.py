import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from PIL import Image
# from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch
from torchvision.utils import save_image 
import cv2

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)


            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        
        return out


class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='small', 
        nclass=21,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
    ):
        super(DPT, self).__init__()
        
        # ==================== 核心修改 ====================
        print("正在从官方 DINOv3 仓库加载 ViT-S+/16 结构...")
        
        # 1. 指向 'facebookresearch/dinov3' (注意是 dinov3 不是 dinov2)
        # 2. 模型名使用 'dinov3_vits16_plus' (对应你下载的 S+ 版本)
        #    如果你下载的是普通 S 版，就用 'dinov3_vits16'
        # 3. pretrained=False (因为你要加载自己下载的权重)
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16plus', pretrained=False)
        except Exception as e:
            print(f"Hub加载失败，请确保网络通畅或检查仓库名: {e}")
            # 如果 hub 连不上，你需要把 dinov3 的代码 clone 下来放到本地引用
            raise e

        # 4. 修正 patch_size (DINOv3 原生就是 16，但我们需要记录这个变量给 DPT 用)
        self.patch_size = 16 
        self.backbone.embed_dim = 384 # S+ 的维度通常还是 384，为了保险手动指定一下给 Head 用
        # ================================================

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }
        
        # Head 部分不需要变
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def forward(self, x, comp_drop=False):
        # 记得把所有的 14 改成 self.patch_size (即 16)
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        
        # DINOv3 的 forward_features 接口可能稍有不同，通常返回 dict
        # 如果报错，可能需要改成 self.backbone.forward_features(x)['x_norm_patchtokens']
        # 但大多数 hub 模型会自动处理 forward
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx['small'] # 强制指定为 small 的索引
        )

        # for idx,feature in enumerate(features):
        #     feature_path = f"./features/0005/{idx+1}.pth"
        #     torch.save(feature, feature_path)

        # for idx, feature in enumerate(features):
        #     # feature=feature.permute(0, 2, 1).reshape((feature.shape[0], feature.shape[-1], patch_h, patch_w))

        #     print(f"feature {idx} shape: {feature.shape}")
        #     feature=feature.squeeze(0)
        #     cv2.imwrite(f"./features/0001/feature_{idx+1}.png", feature.detach().cpu().numpy())




        if comp_drop:
            bs, dim = features[0].shape[0], features[0].shape[-1]
            
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).cuda() * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0
            
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            
            features = (feature * dropout_mask.unsqueeze(1) for feature in features)
            
            out = self.head(features, patch_h, patch_w)
            
            out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
            
            return out
        
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        
        return out
    
if __name__ == 'main':
    x=torch.randn(1,3,518,518).cuda()
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs['small'], 'nclass': 3}).cuda()
    y=model(x)
    print(y.shape)

    


