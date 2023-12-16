from segment_anything.modeling.sam import Sam
import pytorch_lightning as pl
from .lora import *
from segment_anything import sam_model_registry
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
import random

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