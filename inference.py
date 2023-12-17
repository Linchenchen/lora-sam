from lora_sam import *


def inference(checkpoint = "lightning_logs/version_1/checkpoints/epoch=5-step=600.ckpt"):
    plotter = Plotter(checkpoint)
    _, test_loader = get_loaders(batch_size=8)

    input_box = np.array([40, 50, 130, 110])
    images, target = next(iter(test_loader))
    plotter.inference_plot(images[0], target[0], bboxes=input_box)


if __name__ == "__main__":
    inference()