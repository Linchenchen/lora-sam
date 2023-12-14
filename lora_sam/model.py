from segment_anything.modeling.sam import Sam
import pytorch_lightning as pl
from lora import *
from segment_anything import sam_model_registry
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
import random

from .loss_func import MaskDiceLoss, MaskFocalLoss


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

        #self.__apply_lora(sam.image_encoder, lora_rank, lora_scale)
        #BaseFinetuning.freeze(sam.image_encoder, train_bn=True)
        self.sam = sam
        self.dice = MaskDiceLoss()
        self.focal = MaskFocalLoss()
    

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
        kwargs["multimask_output"] = True
        return self.sam(args, kwargs)

    def configure_optimizers(self):
        lora_parameters = [param for param in self.parameters() if param.requires_grad]
        # make sure original sam don't requires_grad
        optimizer = torch.optim.AdamW(lora_parameters, lr=1e-5)
        return optimizer
    

    def calc_loss(self, pred, gt_masks):
        pred_ious  = pred["iou_predictions"][0]
        pred_masks = pred["masks"][0]
        loss = 0
        return self.dice(pred_masks[0], gt_masks[0])
        for iou, pred, targ in zip(pred_ious, pred_masks, gt_masks):
            #loss += 
            return self.dice(pred, targ)
            return self.mask_focal_loss(pred, targ)
            #losses.append(loss)

        return loss
    

    def mask_dice_loss(self, mask_pred, mask_gt):
        mask_pred = mask_pred.to(self.device)
        mask_gt = mask_gt.to(self.device)
        dice_loss = 2 * torch.sum(mask_pred * mask_gt, dim=(-1,-2)) 
        dice_loss /= 1e-5 + torch.sum(mask_pred ** 2, dim=(-1,-2)) + torch.sum(mask_gt ** 2, dim=(-1,-2))
        return torch.sum(1 - dice_loss)
    

    def mask_focal_loss(self, mask_pred: torch.Tensor, mask_gt):
        alpha = 0.8
        gamma = 2
        ids = torch.where(mask_gt == 1)
        focal_loss1 = -alpha * ((~mask_pred[ids]) ** gamma) * torch.log(mask_pred[ids])
        
        ids = torch.where(mask_gt != 1)
        focal_loss2 = -(1-alpha) * (mask_pred[ids] ** gamma) * torch.log(~mask_pred[ids])

        focal_loss = torch.sum(focal_loss1) + torch.sum(focal_loss2)
        focal_loss /= len(mask_pred)
        return focal_loss


    def iou_token_loss(self, iou_prediction, prediction, targets):
        ...


    def point_sample(self, all_masks, points_coords, points_label):
        # all_masks: [N, H, W], one image, N masks
        # points_coords: (N, 2)
        # points_label: (N, 1), 1 for foreground, 0 for background
        # return: sampled_masks: [3, H, W], masks order from big to small
        # you can modify the signature of this function

        mask_ids = []
        for i, mask in enumerate(all_masks):
            is_valid = True
            for is_fore, (x, y) in zip(points_label, points_coords):
                on_mask = int(mask[int(x)][int(y)])
                is_valid = (on_mask and is_fore) or (not on_mask and not is_fore)
                if not is_valid:
                    break

            if is_valid:
                mask_ids.append(i)

        mask_ids.sort(key=lambda i: all_masks[i].sum())
        assert mask_ids

        while len(mask_ids) < 3:
            mask_ids.insert(0, mask_ids[0])

        return mask_ids[:3]
    

    def training_step(self, batch, batch_idx):
        images, target = batch
        images = images.to(self.device)
        target = target.to(self.device)
        
        sam_input = {}
        sam_input['image'] = images[0]
        sam_input['original_size'] = target.shape[2:]
        use_point_prompt = True or torch.rand(1).item() < 0.5

        coords = []
        labels = []
        def append_point(idx, arg):
            i, j = arg
            coords.append(arg)
            labels.append(target[0,idx,i,j])

        idx = torch.randint(target.shape[1], (1,))
        if use_point_prompt:
            append_point(idx, [torch.randint(dim, (1,)) for dim in target.shape[2:]])
            sam_input["point_coords"] = torch.Tensor([coords]).to(self.device)
            sam_input["point_labels"] = torch.Tensor([labels]).to(self.device)
        else:
            pass

        pred = self.forward(sam_input)[0]
        mask_ids = self.point_sample(target[0], coords, labels)
        print("lll2")
        loss = self.calc_loss(pred, target[0][mask_ids])
        print("lll2.5")
        loss.backward()
        print("lll3")


        if use_point_prompt:
            mask_ids = self.point_sample(target[0], coords, labels)
        else:
            pass

        # 1a. single point prompt training
        # 1b. iterative point prompt training up to 3 iteration
        # 2. box prompt training, only 1 iteration
        for _ in range(3):
            pred = self.forward(sam_input)[0]
            print(target[0].shape)
            mask_ids = self.point_sample(target[0], coords, labels)

            for i, mask_idx in enumerate(mask_ids):
                pred_mask = pred["masks"][0][i]
                targ_mask = target[0][mask_idx]

                samp_mask = torch.logical_xor(pred_mask, targ_mask)
                print(samp_mask.shape)
                samp_idx = random.choice(torch.nonzero(samp_mask))
                print(samp_idx)

                append_point(mask_idx, samp_idx)

            logit_id = torch.argmax(pred["iou_predictions"][0], dim=0)
            #sam_input["mask_inputs"] = pred["low_res_logits"][:,logit_id,:,:]
            sam_input["point_coords"] = torch.Tensor([coords]).to(self.device)
            sam_input["point_labels"] = torch.Tensor([labels]).to(self.device)

        print("yattaaaaaaadzbzdb!!!!!!!!")
        pred = self.forward(sam_input)[0]
        print("1")
        mask_ids = self.point_sample(target[0], coords, labels)
        print("2")
        loss = self.calc_loss(pred, target[0][mask_ids])
        print(loss)
        print("3")

        self.log('train_loss', loss, prog_bar=True)
        print("4")
        loss.backward()

        print("yattaaaaaaa!!!!!!!!2222222")
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