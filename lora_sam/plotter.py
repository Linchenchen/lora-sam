import torch
import numpy as np
from .model import LoRASAM
import matplotlib.pyplot as plt
from segment_anything import SamPredictor


class Plotter:
    def __init__(self, checkpoint = "lightning_logs/version_0/checkpoints/epoch=29-step=3000.ckpt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {"lora_rank":4, "lora_scale":1}
        self.sam = LoRASAM.load_from_checkpoint(checkpoint, **kwargs).to(self.device)
        self.predictor = SamPredictor(self.sam.sam)


    def inference_plot(self, img: torch.Tensor, coords=None, labels=None, bboxes=None):
        img.to(self.device)

        masks, scores, logits = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box = None if bboxes is None else bboxes[None, :],
            multimask_output=True,
        )

        print(masks.shape, scores.shape, logits.shape)