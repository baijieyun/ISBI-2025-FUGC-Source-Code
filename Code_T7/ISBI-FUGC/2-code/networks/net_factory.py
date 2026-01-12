from networks.unet import UNet, UNet_2d
import torch.nn as nn

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
    if net_type == "unet" and mode == "train":
        ours_net = UNet(in_chns=in_chns, class_num=class_num).cuda()

    return ours_net

def net(in_chns=3, class_num=3, ema=False):
    net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

