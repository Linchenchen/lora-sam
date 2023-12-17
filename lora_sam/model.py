import pytorch_lightning as pl
from .lora import *
from segment_anything import sam_model_registry
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
import random

from .mask_loss import *


class LoRASAM(pl.LightningModule):
    def __init__(self, lora_rank: int, lora_scale: float, checkpoint="sam_vit_b_01ec64.pth"):
        super().__init__()

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        sam = sam.to(self.device)
        
        sam.image_encoder.img_size = 256

        avg_pooling = nn.AvgPool2d(kernel_size=4, stride=4)
        downsampled_tensor = avg_pooling(sam.image_encoder.pos_embed.permute(0,3,1,2)).permute(0,2,3,1)
        sam.image_encoder.pos_embed.data = downsampled_tensor
        sam.prompt_encoder.input_image_size = [256, 256]
        sam.prompt_encoder.image_embedding_size = [16, 16]

        BaseFinetuning.freeze(sam, train_bn=True)
        self.__apply_lora(sam.image_encoder, lora_rank, lora_scale)
        self.__apply_lora(sam.prompt_encoder, lora_rank, lora_scale)
        self.__apply_lora(sam.mask_decoder, lora_rank, lora_scale)
        
        self.sam = sam
        self.iou_loss = MaskIoULoss()
        self.dice_loss = MaskDiceLoss()
        self.focal_loss = MaskFocalLoss()


    def point_sample(self, points_coords, points_label, all_masks):
        # all_masks: [N, H, W], one image, N masks
        # points_coords: (N, 2)
        # points_label: (N, 1), 1 for foreground, 0 for background
        # return: sampled_masks: [3, H, W], masks order from big to small
        # you can modify the signature of this function
        
        # find the target mask that contains the point
        mask_ids = []
        for i, mask in enumerate(all_masks):
            is_valid = True
            for is_fore, (x, y) in zip(points_label, points_coords):
                on_mask = mask[y][x]
                is_valid = (on_mask and is_fore) or (not on_mask and not is_fore)
                if not is_valid:
                    break

            if is_valid:
                mask_ids.append(i)

        # assign according to the size of the mask and leave one or two of the three empty.
        sampled_masks = torch.zeros((3, all_masks.shape[1], all_masks.shape[2]))
        mask_ids.sort(key=lambda i: all_masks[i].sum(), reverse=True)
        
        for i, idx in enumerate(mask_ids):
            sampled_masks[i] = all_masks[idx]
        
        return sampled_masks


    def box_sample(self, bbox, all_masks):
        # all_masks: [N, H, W], one image, N masks
        # bbox: (xyxy)
        # return: sampled_masks: [3, H, W], masks order from big to small

        # Create a binary mask for the bounding box
        bbox_mask = torch.zeros_like(all_masks)
        bbox_mask[:,bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        # Calculate IoU for all masks
        intersections = torch.logical_and(all_masks, bbox_mask).sum(dim=(1, 2))
        unions = torch.logical_or(all_masks, bbox_mask).sum(dim=(1, 2))
        ious = intersections.float() / unions.float()
        

        # Find the mask indices with the highest IoU and smaller size than bbox
        mask_ids, = torch.where(unions <= bbox_mask[0].sum())

        # Sort by IoU in descending order
        sorted_mask_ids = torch.argsort(ious[mask_ids], descending=True)

        # Assign according to the size of the mask and leave one or two of the three empty.
        sampled_masks = torch.zeros((3, all_masks.size(1), all_masks.size(2)))

        for i, idx in enumerate(sorted_mask_ids[:3]):
            sampled_masks[i] = all_masks[mask_ids[idx]]

        return sampled_masks


    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        images = torch.stack([self.sam.preprocess(img) for img in images], dim=0)
        image_embeddings = self.sam.image_encoder(images)

        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=bbox[None, :],
                masks=None,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious
    

    def configure_optimizers(self):
        lora_parameters = [param for param in self.parameters(recurse=True) if param.requires_grad]

        # make sure original sam don't requires_grad
        optimizer = torch.optim.AdamW(lora_parameters, lr=1e-5)
        return optimizer
    
    
    def random_sample_bbox(self, image_shape):
        H, W = image_shape[-2:]
        # Generate random coordinates for the first point
        x_min, x_max = torch.randint(0, W, (2,)).sort().values
        y_min, y_max = torch.randint(0, H, (2,)).sort().values
        return x_min, y_min, x_max, y_max
        

    def calc_loss(self, batch):
        images, target = batch
        images = images.to(self.device)
        target = [mask.to(self.device) for mask in target]

        target_masks = []
        target_boxes = []
        for i in range(len(images)):
            for _ in range(10):
                bbox = self.random_sample_bbox(images.shape)
                masks = self.box_sample(bbox, target[i])
                if any((mask == 0).all() for mask in masks):
                    continue
                
            target_boxes.append(bbox)
            target_masks.append(masks.to(self.device))

        target_boxes = torch.Tensor(target_boxes).to(self.device)
        mask_preds, iou_preds = self.forward(images, target_boxes)

        focal_loss = torch.tensor(0., device=self.device)
        dice_loss = torch.tensor(0., device=self.device)
        iou_loss = torch.tensor(0., device=self.device)

        for mask_pred, mask_gt, iou_pred in zip(mask_preds, target_masks, iou_preds):
            mask_gt = mask_gt.unsqueeze(0)
            focal_loss += 20. * self.focal_loss(mask_pred, mask_gt)
            dice_loss += self.dice_loss(mask_pred, mask_gt)
            iou_loss += self.iou_loss(iou_pred, mask_pred, mask_gt)

        #print(focal_loss.item(), dice_loss.item(), iou_loss.item())
        
        return focal_loss + dice_loss + iou_loss


    def training_step(self, batch, batch_idx):        
        loss = self.calc_loss(batch)
        self.log('train_loss', loss, prog_bar=True)
        
        # During training, we backprop only the minimum loss over the 3 output masks.
        # sam paper main text Section 3
        return loss
    

    def validation_step(self, batch, batch_idx):
        loss = self.calc_loss(batch)
        # use same procedure as training, monitor the loss
        self.log('val_loss', loss, prog_bar=True)


    def __apply_lora(self, module: nn.Module, r, s):
        for name, blk in module.named_children():
            if isinstance(blk, nn.Linear):
                blk = MonkeyPatchLoRALinear(blk, r, s).to(self.device)
                setattr(module, name, blk)

            elif isinstance(blk, nn.Conv2d):
                blk = MonkeyPatchLoRAConv2D(blk, r, s).to(self.device)
                setattr(module, name, blk)

            elif isinstance(blk, nn.ConvTranspose2d):
                blk = MonkeyPatchLoRAConvTranspose2D(blk, r, s).to(self.device)
                setattr(module, name, blk)

            elif isinstance(blk, nn.Module):
                self.__apply_lora(blk, r, s)