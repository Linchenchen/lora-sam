import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskFocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=0.8):
        super(MaskFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return torch.mean(focal_loss)

    
class MaskDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
    
    
class MaskIoULoss(nn.Module):
    def __init__(self):
        super(MaskIoULoss, self).__init__()
    

    def calc_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor):
        pred_mask = (pred_mask >= 0.5).float()
        intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(-2, -1))
        union = torch.sum(pred_mask, dim=(-2, -1)) + torch.sum(gt_mask, dim=(-2, -1)) - intersection
        epsilon = 1e-7
        batch_iou = intersection / (union + epsilon)

        batch_iou = batch_iou.unsqueeze(1)
        return batch_iou


    def forward(self, iou_pred, mask_pred, mask_gt):
        iou = self.calc_iou(mask_pred, mask_gt)
        return F.mse_loss(iou_pred, iou, reduction='mean')