import torch
import torch.nn.functional as F
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        """
        :param y_pred: one-hot encoded predictions (BS,H,W,C)
        :param y_truth: one-hot encoded predictions
        :return:
        """
        y_pred_f = torch.flatten(y_pred, start_dim=0, end_dim=2)
        y_truth_f = torch.flatten(y_truth, start_dim=0, end_dim=2)
        # print(y_pred_f.shape,y_truth_f.shape)
        dice1 = (2. * ((y_pred_f[:, 1:2] * y_truth_f[:, 1:2]).sum()) + self.smooth) / (
                y_pred_f[:, 1:2].sum() + y_truth_f[:, 1:2].sum() + self.smooth)
        dice2 = (2. * ((y_pred_f[:, 2:] * y_truth_f[:, 2:]).sum()) + self.smooth) / (
                y_pred_f[:, 2:].sum() + y_truth_f[:, 2:].sum() + self.smooth)

        dice1.requires_grad_(True)
        dice2.requires_grad_(True)
        return 1 - (dice1 + dice2) / 2


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
    def forward(self, y_pred, y_truth):
        """
        :param y_pred:   one-hot
        :param y_truth:  one-hot
        :return:
        """
        y_pred_f = torch.flatten(y_pred.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
        target_t = torch.flatten(y_truth, start_dim=0, end_dim=2)
        loss_ce = F.cross_entropy(y_pred_f, target_t.argmax(dim=1))
        return loss_ce

class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.2, 0.35, 0.45], gamma=2, num_classes=3, reduction='mean'):
        """
        初始化 Focal Loss 类。

        参数:
        alpha (float 或 list): 类别权重，用于处理类别不平衡问题。可以是一个标量，也可以是一个长度等于类别数的列表。
        gamma (float): 调节因子，用于降低易分类样本的权重。
        num_classes (int): 类别数量，这里固定为 3 分类任务。
        reduction (str): 损失的缩减方式，可选值为 'none'、'mean' 或 'sum'。
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha] * num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes, f"alpha 列表长度必须等于类别数 {num_classes}"
            self.alpha = torch.tensor(alpha)
        else:
            raise ValueError("alpha 必须是 float、int 或长度等于类别数的列表")
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        """
        前向传播函数，计算 Focal Loss。

        参数:
        y_pred (torch.Tensor): 模型的预测输出，形状为 (N, C, H, W)，其中 N 是批量大小，C 是类别数，H 和 W 是图像的高度和宽度。
        y_truth (torch.Tensor): 真实标签，形状为 (N, C, H, W) 的 one-hot 编码形式。

        返回:
        torch.Tensor: 计算得到的 Focal Loss。
        """
        # 展平预测结果和真实标签，方便后续计算
        preds = torch.flatten(preds.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
        labels = torch.flatten(labels, start_dim=0, end_dim=2)
        labels = labels.argmax(dim=1)
        # preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss