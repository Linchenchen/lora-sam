from lora_sam import *


def inference(checkpoint = "lightning_logs/version_2/checkpoints/epoch=2-step=300.ckpt"):
    plotter = Plotter(checkpoint)
    train_loader, test_loader = get_loaders(batch_size=8)

    input_box = np.array([40, 50, 130, 110])
    input_point = np.array([[50,50], [75,75]])
    input_label = np.array([0, 1])
    kwargs = {}
    #kwargs["bboxes"] = input_box
    kwargs["coords"] = input_point
    kwargs["labels"] = input_label

    
    images, target = next(iter(test_loader))
    plotter.inference_plot(images[0], target[0], **kwargs)


if __name__ == "__main__":
    inference()