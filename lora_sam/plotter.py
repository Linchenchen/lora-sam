import torch
import numpy as np
from .model import LoRASAM
import matplotlib.pyplot as plt
from segment_anything import SamPredictor
import torchvision.transforms as transforms


class Plotter:
    def __init__(self, checkpoint = "lightning_logs/version_0/checkpoints/epoch=29-step=3000.ckpt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {"lora_rank":4, "lora_scale":1}
        self.sam = LoRASAM.load_from_checkpoint(checkpoint, **kwargs).to(self.device)
        self.predictor = SamPredictor(self.sam.sam)
        self.reverse = transforms.ToPILImage()
    

    def __to_numpy(self, img):
        return np.array(self.reverse(img))
    

    def inference_plot(self, img, target, coords=None, labels=None, bboxes=None):
        img = self.__to_numpy(img)
        self.predictor.set_image(img)

        masks, scores, logits = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box = None if bboxes is None else bboxes[None, :],
            multimask_output=True,
        )

        gt_masks = self.sam.box_sample(bboxes, target)

        plt.figure(figsize=(6, 4))
        # Plotting masks on subplots
        for i, (score, mask, gt_mask) in enumerate(zip(scores, masks, gt_masks)):
            ax = plt.subplot(2, 3, i+1)
            plt.imshow(img)
            plt.imshow(mask, alpha=0.5)  # You can adjust the alpha value for transparency
            plt.title("IoU {:.4f}".format(score))

            if bboxes is not None:
                x0, y0 = bboxes[0], bboxes[1]
                w, h = bboxes[2] - bboxes[0], bboxes[3] - bboxes[1]
                ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

        # Plotting ground truth masks on the bottom row
        for i, gt_mask in enumerate(gt_masks):
            plt.subplot(2, 3, i+4)  # Starting from the fourth subplot
            plt.imshow(img)
            plt.imshow(gt_mask, alpha=0.5)  # You can adjust the alpha value for transparency
            plt.title("Ground Truth")  

        # Adjust layout for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()