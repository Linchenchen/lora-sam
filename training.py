import lora_sam
import pytorch_lightning as pl


model = lora_sam.LoRASAM(4, 1)
train_loader, val_loader = lora_sam.get_loaders(batch_size=8)
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=30)
trainer.fit(model, train_loader, val_loader)

def train():
    # Create an instance of your LoRASAM model
    model = lora_sam.LoRASAM(4, 1)

    # Create training and validation data loaders
    train_loader, val_loader = lora_sam.get_loaders(batch_size=8)

    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='model-{epoch:02d}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    # Create a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        gpus=1,  # Set the number of GPUs to use
        max_epochs=30,  # Set the maximum number of epochs
        callbacks=[checkpoint_callback],  # Add the ModelCheckpoint callback
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

# Run the training script
if __name__ == '__main__':
    train()