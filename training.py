import lora_sam
import pytorch_lightning as pl


model = lora_sam.LoRASAM(4, 1)
train_loader, val_loader = lora_sam.get_loaders()
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=30)
trainer.fit(model, train_loader, val_loader)