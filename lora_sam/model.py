from segment_anything.modeling.sam import Sam
import pytorch_lightning as pl
from lora import *
from segment_anything import sam_model_registry


class LoRASAM(pl.LightningModule):
    def __init__(self, lora_rank: int, lora_scale: float, checkpoint="sam_vit_b_01ec64.pth"):
        super().__init__()

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        sam = sam.to(self.device)
        self.__apply_lora(sam.image_encoder, lora_rank, lora_scale)
        self.sam = sam


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


    def construct_batched_input(self, images, targets):
        batched_input = {}
        batched_input['image'] = images
        batched_input['mask_inputs'] = targets

        # Assuming you have a function to get original sizes from images
        batched_input['original_size'] = (256, 160)

        # Other keys may need to be filled based on your model requirements
        # You should adapt this based on the actual keys and values your model expects

        return batched_input
    

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
        return self.lora_sam(args, kwargs)

    def configure_optimizers(self):
        lora_parameters = [param for param in self.parameters() if param.requires_grad]
        # make sure original sam don't requires_grad
        optimizer = torch.optim.AdamW(lora_parameters, lr=1e-5)
        return optimizer
    

    @staticmethod
    def get_sam_original():
        pass

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
        print(images.shape, targets.shape)
        batched_input = self.construct_batched_input(images, targets)

        use_point_prompt = torch.rand(1).item() < 0.5
        if use_point_prompt:
            pass
            
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