#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : UNet_pp.py
# @author      : ahua
# @Start Date  : 2024/3/4 22:52
# @Classes     : UNet++网络
from torch import nn
import torch


class DoubleConv(nn.Module):
    """同UNet定义连续的俩次卷积"""

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        # 俩次卷积
        self.d_conv = nn.Sequential(
            # 相比原论文，这里加入了padding与BN
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.d_conv(x)

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_conv(x)

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output
    

    

class UNetPP(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, features=[64, 128, 256, 512, 1024], deep_supervision=False):
        """

        :param in_channel:
        :param out_channel:
        :param features: 各个采样后对应的通道数
        :param deep_supervision: 是否使用深度监督
        """
        super(UNetPP, self).__init__()

        params = {'in_chns': in_channel,
                  'feature_chns': [64, 128, 256, 512, 1024],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': out_channel,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.params=params
        self.decoder = Decoder(params)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

        self.deep_supervision = deep_supervision

        # 下采样的池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 双线性插值进行上采样，也可以通过ConvTranspose2d或者先ConvTranspose2d后插值实现，这里为了方便直接插值
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 原始UNet的下采样层，每个下采样层的第0层卷积
        self.conv0_0 = DoubleConv(in_channel, features[0])
        self.conv1_0 = DoubleConv(features[0], features[1])
        self.conv2_0 = DoubleConv(features[1], features[2])
        self.conv3_0 = DoubleConv(features[2], features[3])
        self.conv4_0 = DoubleConv(features[3], features[4])

        # 每个下采样层的第一层卷积
        self.conv0_1 = DoubleConv(features[0] + features[1], features[0])
        self.conv1_1 = DoubleConv(features[1] + features[2], features[1])
        self.conv2_1 = DoubleConv(features[2] + features[3], features[2])
        self.conv3_1 = DoubleConv(features[3] + features[4], features[3])

        # 每个下采样层的第二层卷积
        self.conv0_2 = DoubleConv(features[0] * 2 + features[1], features[0])
        self.conv1_2 = DoubleConv(features[1] * 2 + features[2], features[1])
        self.conv2_2 = DoubleConv(features[2] * 2 + features[3], features[2])

        # 每个下采样层的第三层卷积
        self.conv0_3 = DoubleConv(features[0] * 3 + features[1], features[0])
        self.conv1_3 = DoubleConv(features[1] * 3 + features[2], features[1])

        # 每个下采样层的第四层卷积
        self.conv0_4 = DoubleConv(features[0] * 4 + features[1], features[0])

        # 分割头，作者原论文写了深度监督之后还过sigmoid，但是UNet没有sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax=nn.Softmax()
        if self.deep_supervision:
            self.final1 = nn.Conv2d(features[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(features[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(features[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(features[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(features[0], out_channel, kernel_size=1)
    
    def Comp_drop(self,feature):
        bs, dim = feature[0].shape[0], feature[0].shape[-1]
        dropout_mask1 = self.binomial.sample((bs // 2, dim)).cuda() * 2.0
        dropout_mask2 = 2.0 - dropout_mask1
        dropout_prob = 0.5
        num_kept = int(bs // 2 * (1 - dropout_prob))
        kept_indexes = torch.randperm(bs // 2)[:num_kept]
        dropout_mask1[kept_indexes, :] = 1.0
        dropout_mask2[kept_indexes, :] = 1.0
        dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
        feature = feature * dropout_mask.unsqueeze(1)
        return feature


    def forward(self, x,need_fp=False,comp_drop=False):


        x0_0 = self.conv0_0(x)
        if comp_drop:
            x0_0 = self.Comp_drop(x0_0)


        x1_0 = self.conv1_0(self.pool(x0_0))
        if comp_drop:
            x1_0 = self.Comp_drop(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))


        x2_0 = self.conv2_0(self.pool(x1_0))
        if comp_drop:
            x2_0 = self.Comp_drop(x2_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))


        x3_0 = self.conv3_0(self.pool(x2_0))
        if comp_drop:
            x3_0 = self.Comp_drop(x3_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        

        x4_0 = self.conv4_0(self.pool(x3_0))
        if comp_drop:
            x4_0 = self.Comp_drop(x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # 使用深度监督，返回四个分割图
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return [output1, output2, output3, output4]

        else:

            if need_fp:
                # output = self.decoder([x0_0_fp, x1_0_fp, x2_0_fp, x3_0_fp, x4_0_fp])
                return output.chunk(2)

            output = self.final(x0_4)
            # output = self.softmax(output)
            # output2=self.sigmoid(x0_4)
            return output


def test():
    # x = torch.randn(1, 3, 544, 336)
    # from torchsummary import summary
    # model = UNetPP()
    # print(x.shape)
    # print(model(x).shape)
    # print(summary(model,(3,544,336),device='cpu'))



    from torchstat import stat
    import torchvision.models as models
    model = UNetPP()
    stat(model, (3, 224, 224))

if __name__ == '__main__':
    test()
