import os
import numpy as np
import torch
import torch.nn as nn
from utils.metrics_logger import MetricsLogger
from config.config import Config
from models.unet import U_Net  
from models.deeplabv3 import get_deeplabv3
import torch.nn.functional as F
from utils.augmentation import BasicTsf
from utils.dataset import UnlabeledDataSets, LabeledDataSets
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from EBS import ebs, weighted_mse_loss
from utils.criterion import MyCriterion
from models.aux import get_aux
import argparse


def get_alpha(epoch):
    return min(1-1/(epoch+1), 0.99)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

# def get_current_consistency_weight(epoch):
#     # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#     return 0.1 * sigmoid_rampup(epoch, 40)
def get_current_consistency_weight(epoch, max_epoch, lambda_max=0.1, rampup_type="exp"):
    """
    计算一致性损失权重 λ(t)
    :param epoch: 当前 epoch
    :param max_epoch: 总 epoch 数
    :param lambda_max: 一致性损失的最大权重
    :param rampup_type: 选择 ramp-up 策略 ("exp", "linear", "warmup")
    :return: 当前一致性损失的权重
    """
    if rampup_type == "exp":  # 指数增长
        return lambda_max * np.exp(-5 * (1 - epoch / max_epoch) ** 2)
    elif rampup_type == "linear":  # 线性增长
        return lambda_max * (epoch / max_epoch)
    elif rampup_type == "warmup":  # 预热 + 平稳
        return lambda_max if epoch >= max_epoch * 0.4 else 0  # 40% 训练后启用一致性损失
    else:
        raise ValueError("Invalid rampup_type. Choose from ['exp', 'linear', 'warmup']")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train0(args):
    # Load configuration
    config = Config()
    labeled_dir = config.get("paths", "labeled_dir")
    unlabeled_dir = config.get("paths", "unlabeled_dir")
    
    output_dir = config.get("paths", "output_dir")

    lr = config.get('training', 'learning_rate')
    gpu_id = config.get("training", "gpu_id")
    batch_labeled = config.get('training', 'batch_labeled')
    batch_unlabeled = config.get('training', 'batch_unlabeled')
    epochs = config.get("training", "epochs")
    warmup_epochs = config.get("training", "warmup_epoch")

    output_dir = args.output_dir
    aux = args.aux
    mu = args.mu
    b = args.b
    is_beta = args.is_beta

    # get dataloader
    ls_t = os.listdir(os.path.join(labeled_dir, 'images'))
    ls_u = os.listdir(os.path.join(unlabeled_dir, 'images'))
    tsf = BasicTsf()
    dataset_l = LabeledDataSets(labeled_dir, ls_t, tsf)
    dataset_u = UnlabeledDataSets(unlabeled_dir, ls_u, tsf)

    labeled_loader = DataLoader(dataset_l, batch_size=batch_labeled, shuffle=True, num_workers=4)
    unlabeled_loader = DataLoader(dataset_u, batch_size=batch_unlabeled, shuffle=True, num_workers=4)
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model1 = U_Net(n1=16).to(device)
    model2 = U_Net(n1=16).to(device)
    for param in model2.parameters():
        param.requires_grad = False  # Teacher 不训练

    optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()), lr=lr,betas=(0.9, 0.999), weight_decay=0.1)
    # optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr,betas=(0.9, 0.999), weight_decay=0.1)
    cosine_epochs = epochs - warmup_epochs
    # 创建Cosine Annealing调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=cosine_epochs)
    criterion = MyCriterion(epochs+1)  # combined loss function

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    iter_num = 0
    max_iterations = epochs * len(unlabeled_loader)
    print(max_iterations)
    for epoch in range(1, epochs + 1):
        model1.train()
        # Warmup阶段的学习率调整
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr * warmup_factor
            # for param_group in optimizer2.param_groups:
            #     param_group['lr'] = lr * warmup_factor

        for batch_idx, _ in enumerate(range(len(unlabeled_loader))):  # 以无标签数据为主循环
            try:
                image_l, label = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                image_l, label = next(labeled_iter)

            try:
                image_u, image_u_w = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                image_u, image_u_w = next(unlabeled_iter)
                
            print(f"Epoch[{epoch}/{epochs}] | Batch {batch_idx}: ", end="")
            
            image_l = image_l.to(dtype=torch.float32, device=device)  # (BS, 3, 336, 544) 
            label = label.to(device=device)  # (BS,336,544)  int64  0: background 1:ps 2:fh
            image_u = image_u.to(dtype=torch.float32, device=device)  #
            image_u_w = image_u_w.to(dtype=torch.float32, device=device)  #

            # sup loss:
            outputs_l1 = model1(image_l)
            loss1 = criterion(outputs_l1, label, epoch)

            # semi loss:
            outputs_u1 = model1(image_u)
            outputs_u1_t = model2(image_u_w)
            outputs_soft_u1_t = torch.softmax(outputs_u1_t, dim=1)
            outputs_soft_u1 = torch.softmax(outputs_u1, dim=1)
            
            pseudo_supervision1 = weighted_mse_loss(
                outputs_soft_u1, outputs_soft_u1_t.detach(), is_wights=is_beta, mu=mu, b=b)
            consistency_weight = get_current_consistency_weight(epoch, epochs)
            semi_loss = consistency_weight * pseudo_supervision1
            model1_loss = loss1 + semi_loss

            print(f' sup_loss: {loss1.item():.4f} | semi_loss: {semi_loss.item():.4f}|{consistency_weight:.4f}|{pseudo_supervision1.item():.4f}')

            optimizer1.zero_grad()
            model1_loss.backward()
            optimizer1.step()

            iter_num = iter_num + 1
            if epoch > warmup_epochs:
                scheduler1.step()
            # update teacher
            alpha = get_alpha(epoch)
            for student_param, teacher_param in zip(model1.parameters(), model2.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data

        if epoch % 150 ==0:
            torch.save(model1.state_dict(), os.path.join(output_dir, f'unet16_{epoch}.pth'))


def train1(args):
    # Load configuration
    config = Config()
    labeled_dir = config.get("paths", "labeled_dir")
    unlabeled_dir = config.get("paths", "unlabeled_dir")
    
    output_dir = config.get("paths", "output_dir")

    lr = config.get('training', 'learning_rate')
    gpu_id = config.get("training", "gpu_id")
    batch_labeled = config.get('training', 'batch_labeled')
    batch_unlabeled = config.get('training', 'batch_unlabeled')
    epochs = config.get("training", "epochs")
    warmup_epochs = config.get("training", "warmup_epoch")

    output_dir = args.output_dir
    aux = args.aux
    mu = args.mu
    b = args.b
    is_beta = args.is_beta

    # get dataloader
    ls_t = os.listdir(os.path.join(labeled_dir, 'images'))
    ls_u = os.listdir(os.path.join(unlabeled_dir, 'images'))
    tsf = BasicTsf()
    dataset_l = LabeledDataSets(labeled_dir, ls_t, tsf)
    dataset_u = UnlabeledDataSets(unlabeled_dir, ls_u, tsf)

    labeled_loader = DataLoader(dataset_l, batch_size=batch_labeled, shuffle=True, num_workers=4)
    unlabeled_loader = DataLoader(dataset_u, batch_size=batch_unlabeled, shuffle=True, num_workers=4)
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    # logger = MetricsLogger(save_dir=output_dir)

    # Initialize model
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model1 = U_Net(n1=16).to(device)
    model2 = get_aux(aux).to(device)

    optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()), lr=lr,betas=(0.9, 0.999), weight_decay=0.1)
    optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr,betas=(0.9, 0.999), weight_decay=0.1)
    cosine_epochs = epochs - warmup_epochs
    # 创建Cosine Annealing调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=cosine_epochs)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=cosine_epochs)

    criterion = MyCriterion(epochs+1)  # combined loss function
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(3) 
    # consistency loss 
    criterion_mse = nn.MSELoss()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    iter_num = 0
    max_iterations = epochs * len(unlabeled_loader)
    print(max_iterations)
    for epoch in range(1, epochs + 1):
        model1.train()
        model2.train()
        # Warmup阶段的学习率调整
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr * warmup_factor
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr * warmup_factor

        for batch_idx, _ in enumerate(range(len(unlabeled_loader))):  # 以无标签数据为主循环
            try:
                image_l, label = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                image_l, label = next(labeled_iter)

            try:
                image_u, image_u_w = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                image_u, image_u_w = next(unlabeled_iter)
                
            print(f"Epoch[{epoch}/{epochs}] | Batch {batch_idx}: ", end="")
            
            image_l = image_l.to(dtype=torch.float32, device=device)  # (BS, 3, 336, 544) 
            label = label.to(device=device)  # (BS,336,544)  int64  0: background 1:ps 2:fh
            image_u = image_u.to(dtype=torch.float32, device=device)  #
            image_u_w = image_u_w.to(dtype=torch.float32, device=device)  #

            # sup loss:
            outputs_l1 = model1(image_l)
            # outputs_soft_l1 = torch.softmax(outputs_l1, dim=1)
            outputs_l2 = model2(image_l)
            # outputs_soft_l2 = torch.softmax(outputs_l2, dim=1)
            
            loss1 = criterion(outputs_l1, label, epoch)
            loss2 = criterion(outputs_l2, label, epoch)

            # semi loss:
            outputs_u1 = model1(image_u)
            outputs_soft_u1 = torch.softmax(outputs_u1, dim=1)
            
            outputs_u2 = model2(image_u)
            outputs_soft_u2 = torch.softmax(outputs_u2, dim=1)
            
            # pseudo_output = ebs(outputs_soft_u1, outputs_soft_u2)

            # Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer

            pseudo_outputs1 = torch.argmax(
                outputs_soft_u1.detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft_u2.detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft_u1, pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft_u2, pseudo_outputs1.unsqueeze(1))
            
            # pseudo_supervision1 = weighted_mse_loss(
            #     outputs_soft_u1, outputs_soft_u2.detach(), is_wights=is_beta, mu=mu, b=b)
            # pseudo_supervision2 = weighted_mse_loss(
            #     outputs_soft_u2, outputs_soft_u1.detach(), is_wights=is_beta, mu=mu, b=b)

            consistency_weight = get_current_consistency_weight(epoch, epochs)
            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            print(f' loss1: {model1_loss.item():.4f} || loss2: {model2_loss.item():.4f}')
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss = model1_loss + model2_loss
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1
            if epoch > warmup_epochs:
                scheduler1.step()
                scheduler2.step()
        if epoch % 150 ==0:
            torch.save(model1.state_dict(), os.path.join(output_dir, f'unet16_{epoch}.pth'))
            # torch.save(model2.state_dict(), os.path.join(output_dir, f'{aux}_{epoch}.pth'))

def train2(args):
    # Load configuration
    config = Config()
    labeled_dir = config.get("paths", "labeled_dir")
    unlabeled_dir = config.get("paths", "unlabeled_dir")
    
    output_dir = config.get("paths", "output_dir")

    lr = config.get('training', 'learning_rate')
    gpu_id = config.get("training", "gpu_id")
    batch_labeled = config.get('training', 'batch_labeled')
    batch_unlabeled = config.get('training', 'batch_unlabeled')
    epochs = config.get("training", "epochs")
    warmup_epochs = config.get("training", "warmup_epoch")

    output_dir = args.output_dir
    aux = args.aux
    mu = args.mu
    b = args.b
    is_beta = args.is_beta

    # get dataloader
    ls_t = os.listdir(os.path.join(labeled_dir, 'images'))
    ls_u = os.listdir(os.path.join(unlabeled_dir, 'images'))
    tsf = BasicTsf()
    dataset_l = LabeledDataSets(labeled_dir, ls_t, tsf)
    dataset_u = UnlabeledDataSets(unlabeled_dir, ls_u, tsf)

    labeled_loader = DataLoader(dataset_l, batch_size=batch_labeled, shuffle=True, num_workers=4)
    unlabeled_loader = DataLoader(dataset_u, batch_size=batch_unlabeled, shuffle=True, num_workers=4)
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    # logger = MetricsLogger(save_dir=output_dir)

    # Initialize model
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model1 = U_Net(n1=16).to(device)
    model1_ema = U_Net(n1=16).to(device)

    # model2 = get_deeplabv3().to(device)
    # model2_ema = get_deeplabv3().to(device)
    model2 = get_aux(aux).to(device)
    model2_ema = get_aux(aux).to(device)

    for param in model1_ema.parameters():
        param.requires_grad = False  # Teacher 不训练
    for param in model2_ema.parameters():
        param.requires_grad = False  # Teacher 不训练
    
    # optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    # optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()), lr=lr,betas=(0.9, 0.999), weight_decay=0.1)
    optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr,betas=(0.9, 0.999), weight_decay=0.1)
    cosine_epochs = epochs - warmup_epochs
    # 创建Cosine Annealing调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=cosine_epochs)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=cosine_epochs)
    # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.98)
    # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.98)
    criterion = MyCriterion(epochs+1)  # combined loss function
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(3) 
    # consistency loss 
    criterion_mse = nn.MSELoss()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    iter_num = 0
    max_iterations = epochs * len(unlabeled_loader)
    print(max_iterations)
    for epoch in range(1, epochs + 1):
        model1.train()
        model2.train()
        model1_ema.train()
        model2_ema.train()
        # Warmup阶段的学习率调整
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr * warmup_factor
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr * warmup_factor

        for batch_idx, _ in enumerate(range(len(unlabeled_loader))):  # 以无标签数据为主循环
            try:
                image_l, label = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                image_l, label = next(labeled_iter)

            try:
                image_u, image_u_w = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                image_u, image_u_w = next(unlabeled_iter)
                
            print(f"Epoch[{epoch}/{epochs}] | Batch {batch_idx}: ", end="")
            
            image_l = image_l.to(dtype=torch.float32, device=device)  # (BS, 3, 336, 544) 
            label = label.to(device=device)  # (BS,336,544)  int64  0: background 1:ps 2:fh
            image_u = image_u.to(dtype=torch.float32, device=device)  #
            image_u_w = image_u_w.to(dtype=torch.float32, device=device)  #

            # sup loss:
            outputs_l1 = model1(image_l)
            # outputs_soft_l1 = torch.softmax(outputs_l1, dim=1)
            outputs_l2 = model2(image_l)
            # outputs_soft_l2 = torch.softmax(outputs_l2, dim=1)
            
            loss1 = criterion(outputs_l1, label, epoch)
            loss2 = criterion(outputs_l2, label, epoch)
            # ce, dice = ce_loss(outputs_l1, label), dice_loss(outputs_soft_l1, label.unsqueeze(1))
            # loss1 = 0.5 * ( ce + dice)
            # loss2 = 0.5 * (ce_loss(outputs_l2, label) + dice_loss(
            #     outputs_soft_l2, label.unsqueeze(1)))

            # semi loss:
            outputs_u1 = model1(image_u)
            outputs_u1_t = model1_ema(image_u_w)
            outputs_soft_u1_t = torch.softmax(outputs_u1_t, dim=1)
            outputs_soft_u1 = torch.softmax(outputs_u1, dim=1)
            
            outputs_u2 = model2(image_u)
            outputs_u2_t = model2_ema(image_u_w)
            outputs_soft_u2_t = torch.softmax(outputs_u2_t, dim=1)
            outputs_soft_u2 = torch.softmax(outputs_u2, dim=1)
            
            pseudo_outputs1 = torch.argmax(
                outputs_soft_u1_t.detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft_u2_t.detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft_u1, pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft_u2, pseudo_outputs1.unsqueeze(1))
            
            # pseudo_supervision1 = weighted_mse_loss(
            #     outputs_soft_u1, pseudo_output.detach(), is_wights=is_beta, mu=mu, b=b)
            # pseudo_supervision2 = weighted_mse_loss(
            #     outputs_soft_u2, pseudo_output.detach(), is_wights=is_beta, mu=mu, b=b)

            consistency_weight = get_current_consistency_weight(epoch, epochs)
            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            print(f' loss1: {model1_loss.item():.4f} || loss2: {model2_loss.item():.4f}')
            # print(f'ce:{ce.item():.4f} | dice:{dice.item():.4f} | loss1: {model1_loss.item():.4f}')

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss = model1_loss + model2_loss
            total_loss.backward()
            # model1_loss.backward()
            # model2_loss.backward()
            optimizer1.step()
            optimizer2.step()

            # update teacher
            alpha = get_alpha(epoch)
            for student_param, teacher_param in zip(model1.parameters(), model1_ema.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
            for student_param, teacher_param in zip(model2.parameters(), model2_ema.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
            

            iter_num = iter_num + 1

            # lr_ = lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer1.param_groups:
            #     param_group['lr'] = lr_
            # for param_group in optimizer2.param_groups:
            #     param_group['lr'] = lr_
                    # # Cosine Annealing阶段的学习率调整
            if epoch > warmup_epochs:
                scheduler1.step()
                scheduler2.step()
        if epoch % 150 ==0:
            torch.save(model1.state_dict(), os.path.join(output_dir, f'unet16_{epoch}.pth'))
            torch.save(model2.state_dict(), os.path.join(output_dir, f'{aux}_aux_{epoch}.pth'))


def train34(args):
    # Load configuration
    config = Config()
    labeled_dir = config.get("paths", "labeled_dir")
    unlabeled_dir = config.get("paths", "unlabeled_dir")
    
    output_dir = config.get("paths", "output_dir")

    lr = config.get('training', 'learning_rate')
    gpu_id = config.get("training", "gpu_id")
    batch_labeled = config.get('training', 'batch_labeled')
    batch_unlabeled = config.get('training', 'batch_unlabeled')
    epochs = config.get("training", "epochs")
    warmup_epochs = config.get("training", "warmup_epoch")

    output_dir = args.output_dir
    aux = args.aux
    mu = args.mu
    b = args.b
    is_beta = args.is_beta

    # get dataloader
    ls_t = os.listdir(os.path.join(labeled_dir, 'images'))
    ls_u = os.listdir(os.path.join(unlabeled_dir, 'images'))
    tsf = BasicTsf()
    dataset_l = LabeledDataSets(labeled_dir, ls_t, tsf)
    dataset_u = UnlabeledDataSets(unlabeled_dir, ls_u, tsf)

    labeled_loader = DataLoader(dataset_l, batch_size=batch_labeled, shuffle=True, num_workers=4)
    unlabeled_loader = DataLoader(dataset_u, batch_size=batch_unlabeled, shuffle=True, num_workers=4)
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    # logger = MetricsLogger(save_dir=output_dir)

    # Initialize model
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model1 = U_Net(n1=16).to(device)
    model1_ema = U_Net(n1=16).to(device)

    # model2 = get_deeplabv3().to(device)
    # model2_ema = get_deeplabv3().to(device)
    model2 = get_aux(aux).to(device)
    model2_ema = get_aux(aux).to(device)

    for param in model1_ema.parameters():
        param.requires_grad = False  # Teacher 不训练
    for param in model2_ema.parameters():
        param.requires_grad = False  # Teacher 不训练
    
    # optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    # optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()), lr=lr,betas=(0.9, 0.999), weight_decay=0.1)
    optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr,betas=(0.9, 0.999), weight_decay=0.1)
    cosine_epochs = epochs - warmup_epochs
    # 创建Cosine Annealing调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=cosine_epochs)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=cosine_epochs)
    # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.98)
    # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.98)
    criterion = MyCriterion(epochs+1)  # combined loss function
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(3) 
    # consistency loss 
    criterion_mse = nn.MSELoss()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    iter_num = 0
    max_iterations = epochs * len(unlabeled_loader)
    print(max_iterations)
    for epoch in range(1, epochs + 1):
        model1.train()
        model2.train()
        model1_ema.train()
        model2_ema.train()
        # Warmup阶段的学习率调整
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr * warmup_factor
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr * warmup_factor

        for batch_idx, _ in enumerate(range(len(unlabeled_loader))):  # 以无标签数据为主循环
            try:
                image_l, label = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                image_l, label = next(labeled_iter)

            try:
                image_u, image_u_w = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                image_u, image_u_w = next(unlabeled_iter)
                
            print(f"Epoch[{epoch}/{epochs}] | Batch {batch_idx}: ", end="")
            
            image_l = image_l.to(dtype=torch.float32, device=device)  # (BS, 3, 336, 544) 
            label = label.to(device=device)  # (BS,336,544)  int64  0: background 1:ps 2:fh
            image_u = image_u.to(dtype=torch.float32, device=device)  #
            image_u_w = image_u_w.to(dtype=torch.float32, device=device)  #

            # sup loss:
            outputs_l1 = model1(image_l)
            # outputs_soft_l1 = torch.softmax(outputs_l1, dim=1)
            outputs_l2 = model2(image_l)
            # outputs_soft_l2 = torch.softmax(outputs_l2, dim=1)
            
            loss1 = criterion(outputs_l1, label, epoch)
            loss2 = criterion(outputs_l2, label, epoch)
            # ce, dice = ce_loss(outputs_l1, label), dice_loss(outputs_soft_l1, label.unsqueeze(1))
            # loss1 = 0.5 * ( ce + dice)
            # loss2 = 0.5 * (ce_loss(outputs_l2, label) + dice_loss(
            #     outputs_soft_l2, label.unsqueeze(1)))

            # semi loss:
            outputs_u1 = model1(image_u)
            outputs_u1_t = model1_ema(image_u_w)
            outputs_soft_u1_t = torch.softmax(outputs_u1_t, dim=1)
            outputs_soft_u1 = torch.softmax(outputs_u1, dim=1)
            
            outputs_u2 = model2(image_u)
            outputs_u2_t = model2_ema(image_u_w)
            outputs_soft_u2_t = torch.softmax(outputs_u2_t, dim=1)
            outputs_soft_u2 = torch.softmax(outputs_u2, dim=1)
            
            pseudo_output = ebs(outputs_soft_u1_t, outputs_soft_u2_t)


            # pseudo_supervision1 = criterion_mse(
            #     outputs_u1, outputs_u1_t.detach())
            # pseudo_supervision2 = criterion_mse(
            #     outputs_u2, pseudo_output.detach())
            pseudo_supervision1 = weighted_mse_loss(
                outputs_soft_u1, pseudo_output.detach(), is_wights=is_beta, mu=mu, b=b)
            pseudo_supervision2 = weighted_mse_loss(
                outputs_soft_u2, pseudo_output.detach(), is_wights=is_beta, mu=mu, b=b)

            consistency_weight = get_current_consistency_weight(epoch, epochs)
            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            print(f' loss1: {model1_loss.item():.4f} || loss2: {model2_loss.item():.4f}')
            # print(f'ce:{ce.item():.4f} | dice:{dice.item():.4f} | loss1: {model1_loss.item():.4f}')

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss = model1_loss + model2_loss
            total_loss.backward()
            # model1_loss.backward()
            # model2_loss.backward()
            optimizer1.step()
            optimizer2.step()

            # update teacher
            alpha = get_alpha(epoch)
            for student_param, teacher_param in zip(model1.parameters(), model1_ema.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
            for student_param, teacher_param in zip(model2.parameters(), model2_ema.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
            

            iter_num = iter_num + 1

            # lr_ = lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer1.param_groups:
            #     param_group['lr'] = lr_
            # for param_group in optimizer2.param_groups:
            #     param_group['lr'] = lr_
                    # # Cosine Annealing阶段的学习率调整
            if epoch > warmup_epochs:
                scheduler1.step()
                scheduler2.step()
        if epoch % 150 ==0:
            torch.save(model1.state_dict(), os.path.join(output_dir, f'unet16_{epoch}.pth'))
            torch.save(model2.state_dict(), os.path.join(output_dir, f'{aux}_aux_{epoch}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSMIS")
    parser.add_argument("--aux", type=str, default="deeplabv3_resnet50")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=0.5)
    parser.add_argument("--mode", type=int, default=3)
    parser.add_argument("--is_beta", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save results")

    args = parser.parse_args()
    if args.mode == 0:
        train0(args)
    elif args.mode == 1:
        train1(args)
    elif args.mode == 2:
        train2(args)
    else :
        train34(args)
