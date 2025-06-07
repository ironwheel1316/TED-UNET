import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5, ignore_index=None):
        """
        Focal Loss for multi-class classification, which focuses training on hard, misclassified examples.
        Args:
            alpha (float, optional): Balancing factor. Default is 1.
            gamma (float, optional): Focusing parameter. Default is 2.
            ignore_index (int, optional): Specifies a target class index to ignore.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index


    def forward(self, logits, target_one_hot):
        """
        Args:
            logits (torch.Tensor): Model output, shape (N, C, H, W).
            target_one_hot (torch.Tensor): Ground truth, shape (N, C, H, W),
                                           where each channel is a binary mask (0.0 or 1.0).
        Returns:
            torch.Tensor: Calculated focal loss.
        """
       
        # Convert one-hot-like target to class indices
        target_indices = torch.argmax(target_one_hot, dim=1)
        
        # Compute softmax values for each class
        ce_loss = F.cross_entropy(logits, target_indices, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
