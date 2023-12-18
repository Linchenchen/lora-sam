from lora_sam import *


def inference(checkpoint = "lightning_logs/version_6/checkpoints/epoch=24-step=27975.ckpt"):
    input_transform = transforms.Compose([
    transforms.Resize((160, 256), antialias=True),
    transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 256), antialias=True),
    ])

    plotter = Plotter(checkpoint)
    dataset = SA1B_Dataset("./data", transform=input_transform, target_transform=target_transform)
    #train_loader, test_loader = get_loaders(batch_size=8)

    image, target = dataset.__getitem__(158)

    input_box = np.array([40, 50, 130, 110])
    input_point = np.array([[50,50], [125,125]])
    input_label = np.array([0, 1])
    kwargs = {}
    # kwargs["bboxes"] = input_box
    kwargs["coords"] = input_point
    kwargs["labels"] = input_label

    
    # images, target = next(iter(test_loader))
    plotter.inference_plot(image, target, **kwargs)


if __name__ == "__main__":
    inference()