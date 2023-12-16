import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.8):
        super(MaskFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return torch.mean(focal_loss)

    
class MaskDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(MaskDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() + self.smooth
        dice_loss = 1 - (2 * intersection + self.smooth) / union
        return dice_loss
    
    
class MaskIoULoss(nn.Module):
    def __init__(self):
        super(MaskIoULoss, self).__init__()
    

    def calc_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor):
        pred_mask = (pred_mask >= 0.5).float()
        intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
        union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
        epsilon = 1e-7
        batch_iou = intersection / (union + epsilon)

        batch_iou = batch_iou.unsqueeze(1)
        return batch_iou


    def forward(self, iou_pred, mask_pred, mask_gt):
        iou = self.calc_iou(mask_pred, mask_gt)
        return F.mse_loss(iou_pred, iou, reduction='mean')