from torch import nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    """Multiclass Squared Dice Loss (Generalized Dice Loss) for better handling of imbalanced segmentation.
    
    Reference: 
    - https://arxiv.org/pdf/1707.03237.pdf (Generalized Dice Loss)
    """

    def __init__(self, num_classes, softmax_dim=1, ignore_index=None):
        """
        Args:
            num_classes (int): Number of classes.
            softmax_dim (int): Dimension to apply softmax (default: 1, for channels).
            ignore_index (int or None): If set, this class index will be ignored in loss computation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim
        self.ignore_index = ignore_index

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """
        Compute the Squared Dice Loss for multi-class segmentation.

        Args:
            logits (Tensor): Predicted logits of shape (N, C, H, W).
            targets (Tensor): Ground truth labels of shape (N, H, W) with class indices.
            reduction (str): 'mean', 'sum', or 'none' (default: 'mean').
            smooth (float): Smoothing factor to prevent division by zero.

        Returns:
            Tensor: Squared Dice loss value.
        """

        # Apply softmax to get class probabilities
        probabilities = F.softmax(logits, dim=self.softmax_dim)

        # Convert targets to one-hot encoding: (N, H, W) → (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Ignore specific class index if provided
        if self.ignore_index is not None:
            mask = targets != self.ignore_index  # Create a mask where ignored pixels are False
            targets_one_hot = targets_one_hot * mask.unsqueeze(1)  # Apply mask to one-hot targets
            probabilities = probabilities * mask.unsqueeze(1)  # Apply mask to predictions

        # Compute intersection and union per class (SQUARED TERMS)
        intersection = ((targets_one_hot * probabilities).sum(dim=(2, 3)))**2  # Squared Intersection
        union = ((targets_one_hot + probabilities).sum(dim=(2, 3)))**2  # Squared Denominator

        # Compute per-class Squared Dice score
        dice_coeff = (2. * intersection + smooth) / (union + smooth)

        # Compute Dice loss (1 - Squared Dice coefficient)
        dice_loss = 1 - dice_coeff

        # Apply reduction
        if reduction == 'mean':
            return dice_loss.mean()  # Average over batch and classes
        elif reduction == 'sum':
            return dice_loss.sum()  # Sum over batch and classes
        else:
            return dice_loss  # Return per-class loss tensor