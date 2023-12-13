from lora_sam import SA1B_Dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split



def get_loaders(folder="./sa1b"):
    input_transform = transforms.Compose([
        transforms.Resize((160, 256)),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 256)),
    ])

    dataset = SA1B_Dataset(folder, transform=input_transform, target_transform=target_transform)

    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    torch.random.manual_seed(1)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, test_loader