import numpy as np
import torch
import albumentations as A
import matplotlib.pyplot as plt

inv_normalise = A.Compose([
    A.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        max_pixel_value=255
    )])


def display_images(im):
    seq = im['seq'].cpu().permute(0, 1, 3, 4, 2).numpy()
    gts = im['gt1_seq'].cpu().permute(0, 1, 3, 4, 2).numpy()
    fig = plt.figure(f" Sample images", figsize=(20, 16))
    plt.axis("off")
    for x, i in enumerate(range(0, 18, 6)):
        # create a subplot
        # seq shape = B, T, C, H, W

        for frame in range(0, 3):
            plt.subplot(6, 3, i + 1 + frame)
            denorm = inv_normalise(image=seq[x][frame] * 255)
            plt.imshow(denorm['image'])
        for frame in range(0, 3):
            plt.subplot(6, 3, i + 1 + 3 + frame)
            plt.imshow(gts[x][frame].astype("uint8"))

        # image = image.transpose((1, 2, 0))

        # grab the label id and get the label from the classes list
        # idx = batch[1][i]
        # label = classes[idx]
        # show the image along with the label
        # plt.imshow(image)
        # plt.title(label)

    # show the plot
    plt.tight_layout()
    plt.show()
