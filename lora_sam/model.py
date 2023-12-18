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

        # self.__apply_lora(sam.image_encoder, lora_rank, lora_scale)
        # BaseFinetuning.freeze(sam.image_encoder, train_bn=True)
        self.sam = sam
        self.iou_loss = MaskIoULoss()
        self.dice_loss = MaskDiceLoss()
        self.focal_loss = MaskFocalLoss()


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
                multimask_output=False,
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
        lora_parameters = [param for param in self.parameters() if param.requires_grad]
        # make sure original sam don't requires_grad
        optimizer = torch.optim.AdamW(lora_parameters, lr=1e-5)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        images, target = batch
        images = images.to(self.device)
        target = [mask.to(self.device) for mask in target]

        target_masks = [random.choice(masks).unsqueeze(0) for masks in target]

        bboxes = []
        for mask in target_masks:
            idx = torch.nonzero(mask)
            x_min = torch.min(idx[1])
            y_min = torch.min(idx[0])
            x_max = torch.max(idx[1])
            y_max = torch.max(idx[0])
            bboxes.append((x_min, y_min, x_max, y_max))

        bboxes = torch.Tensor(bboxes).to(self.device)

        loss_focal = torch.tensor(0., device=self.device)
        loss_dice = torch.tensor(0., device=self.device)
        loss_iou = torch.tensor(0., device=self.device)
        
        mask_preds, iou_preds = self.forward(images, bboxes)
            
        for mask_pred, mask_gt, iou_pred in zip(mask_preds, target_masks, iou_preds):
            loss_focal += self.focal_loss(mask_pred, mask_gt)
            loss_dice += self.dice_loss(mask_pred, mask_gt)
            loss_iou += self.iou_loss(iou_pred, mask_pred, mask_gt)
        
        loss = 20. * loss_focal + loss_dice + loss_iou
        self.log('train_loss', loss, prog_bar=True)
        
        # During training, we backprop only the minimum loss over the 3 output masks.
        # sam paper main text Section 3
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        ...
        loss = 0
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

from segment_anything.modeling.sam import Sam
import pytorch_lightning as pl


class MyFastSAM(pl.LightningModule):
    def __init__(self, orig_sam: Sam, lora_rank: int, lora_scale: float):
        super().__init__()
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        sam = sam.to(self.device)
        
        sam.image_encoder.img_size = 256

        avg_pooling = nn.AvgPool2d(kernel_size=4, stride=4)
        downsampled_tensor = avg_pooling(sam.image_encoder.pos_embed.permute(0,3,1,2)).permute(0,2,3,1)
        sam.image_encoder.pos_embed.data = downsampled_tensor
        sam.prompt_encoder.input_image_size = [256, 256]
        sam.prompt_encoder.image_embedding_size = [16, 16]

        # self.__apply_lora(sam.image_encoder, lora_rank, lora_scale)
        # BaseFinetuning.freeze(sam.image_encoder, train_bn=True)
        self.lora_sam = sam
        self.iou_loss = MaskIoULoss()
        self.dice_loss = MaskDiceLoss()
        self.focal_loss = MaskFocalLoss()


    def forward(self, *args, **kwargs):
        """
        comments imported from original SAM code

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        return self.lora_sam(...)

    def configure_optimizers(self):
        lora_parameters = [param for param in self.parameters() if param.requires_grad]
        # make sure original sam don't requires_grad
        optimizer = torch.optim.AdamW(lora_parameters, lr=1e-5)
        return optimizer

    @staticmethod
    def mask_dice_loss(prediction, targets):
        ...

    @staticmethod
    def mask_focal_loss(prediction, targets):
        ...

    @staticmethod
    def iou_token_loss(iou_prediction, prediction, targets):
        ...

    def training_step(self, batch, batch_idx):
        images, targets = batch
        batched_input = self.construct_batched_input(...)
        # 1a. single point prompt training
        # 1b. iterative point prompt training up to 3 iteration
        # 2. box prompt training, only 1 iteration
        predictions = self.forward(batched_input)
        loss = ...
        self.log('train_loss', loss, prog_bar=True)
        # During training, we backprop only the minimum loss over the 3 output masks.
        # sam paper main text Section 3
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        ...
        loss = ...
        # use same procedure as training, monitor the loss
        self.log('val_loss', loss, prog_bar=True)