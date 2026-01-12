import numpy as np
import torch.nn as nn
import torch
import SimpleITK as sitk


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, is_drop=False, n1=16, is_aux=False):
        super(U_Net, self).__init__()
        self.is_drop = is_drop
        n1 = n1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.is_aux = is_aux

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.1)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  # out  #(B,1024,32,32)
        if self.is_drop:
            e5 = self.dropout1(e5)
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        if self.is_drop:
            d4 = self.dropout1(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        if self.is_drop:
            d2 = self.dropout3(d2)

        seg = self.Conv(d2)

        if self.is_aux:
            return seg, d3, d4, d5
        else:
            return seg

class CBR(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 3, kernel_size=3)
        self.bn  = nn.BatchNorm2d(3)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
import torch.nn.functional as F
class ProjectHead(nn.Module):
    def __init__(self, in_c, scale):
        super().__init__()
        self.cbr = CBR(in_c)
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.scale = scale
    def forward(self, x):
        x = self.cbr(x)
        x = F.interpolate(x, scale_factor=self.scale, mode ='bilinear', align_corners=True)
        return self.conv(x)

class Aux(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = U_Net(n1=8, is_aux=True)
        self.p1 = ProjectHead(16, 2)
        self.p2 = ProjectHead(32, 4)
        self.p3 = ProjectHead(64, 8)
    def forward(self, x):
        x, x1, x2, x3 = self.model(x)
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        return x, x1, x2, x3 
if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
    import torch

    # x = torch.randn((1, 3, 336, 544)).cpu()

    # model = U_Net(n1=16, is_aux=True).cpu()
    model = Aux().cpu()
    # print('model parameters: ', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
    # flops = FlopCountAnalysis(model, x)
    # acts = ActivationCountAnalysis(model, x)
    # print(f"total flops : {(flops.total() + acts.total()) / 1e9}", 'G')
    x = torch.rand((2,3,336,544))
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    