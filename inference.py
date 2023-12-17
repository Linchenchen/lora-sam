# import pytorch_lightning as pl
# from lora_sam import *
# from segment_anything import sam_model_registry
# from pytorch_lightning.callbacks.finetuning import BaseFinetuning
# import random

# sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
# sam = sam.to(self.device)

# sam.image_encoder.img_size = 256

# avg_pooling = nn.AvgPool2d(kernel_size=4, stride=4)
# downsampled_tensor = avg_pooling(sam.image_encoder.pos_embed.permute(0,3,1,2)).permute(0,2,3,1)
# sam.image_encoder.pos_embed.data = downsampled_tensor
# sam.prompt_encoder.input_image_size = [256, 256]
# sam.prompt_encoder.image_embedding_size = [16, 16]

# BaseFinetuning.freeze(sam, train_bn=True)
# self.__apply_lora(sam.image_encoder, lora_rank, lora_scale)

# # Define the path to the checkpoint file
# checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=29-step=3000.ckpt"

# # Load the model from the checkpoint
# model = sam.load_from_checkpoint(checkpoint_path)

# # Move the model to the desired device (e.g., GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Set the model to evaluation mode
# model.eval()

import pandas as pd
import matplotlib.pyplot as plt

# Load metrics.csv file
metrics_file = 'lightning_logs/version_1/metrics.csv'
metrics_df = pd.read_csv(metrics_file)

# Group by epoch and calculate the mean
average_metrics_df = metrics_df.groupby('epoch').mean().reset_index()

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(average_metrics_df['epoch'], average_metrics_df['train_loss'], label='Average Train Loss', marker='o')
plt.plot(average_metrics_df['epoch'], average_metrics_df['val_loss'], label='Average Validation Loss', marker='o')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training and Validation Losses')
plt.legend()

# Save the plot
plt.savefig('average_training_validation_losses.png')

# Show the plot
plt.show()
