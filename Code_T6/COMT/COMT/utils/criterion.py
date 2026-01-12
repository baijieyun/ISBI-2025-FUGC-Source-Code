from .loss import *


class MyCriterion(nn.Module):
    def __init__(self, epoch_edge):
        super(MyCriterion, self).__init__()
        self.DC = DiceLoss()
        self.focal = FocalLoss()
        self.ce = CELoss()
        self.epoch_edge = epoch_edge

    def forward(self, pred, label, epoch):
        """

        :param pred: (BS,3,336,544)
        :param label: (BS,336,544)
        :return:
        """
        if epoch >= self.epoch_edge:
            label_onehot = F.one_hot(label.long(), 3) # (BS, 336, 544, 3)
            pred_t = F.one_hot(pred.argmax(dim=1).long(), 3) # (BS, 336, 544, 3)
            dice_loss = self.DC(pred_t, label_onehot)
            focal_loss = self.focal(pred, label_onehot)
            print(f" DC: {dice_loss.detach().cpu().numpy():.5f} || FOCAL: {focal_loss.detach().cpu().numpy():.5f}")
            supervised_loss = 0.5 * focal_loss + 0.5 * dice_loss
            return supervised_loss
        else:
            label_onehot = F.one_hot(label.long(), 3) # (BS, 336, 544, 3)
            pred_t = F.one_hot(pred.argmax(dim=1).long(), 3) # (BS, 336, 544, 3)
            dice_loss = self.DC(pred_t, label_onehot)
            ce_loss = self.ce(pred, label_onehot)
            print(f" DC: {dice_loss.detach().cpu().numpy():.5f} || CE: {ce_loss.detach().cpu().numpy():.5f}", end='')
            supervised_loss = 0.5 * ce_loss + 0.5 * dice_loss
            return supervised_loss


class ClsCriterion(nn.Module):
    def __init__(self):
        super(ClsCriterion, self).__init__()

    def forward(self, pred, label):
        ce_loss = F.cross_entropy(pred, label)
        print(f" CE: {ce_loss.detach().cpu().numpy():.4f}")
        return ce_loss