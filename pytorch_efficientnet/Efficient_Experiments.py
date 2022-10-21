
import torch.nn as nn
from pytorch_efficientnet.efficientnet_v2_seg import EfficientNetV2
import matplotlib.pyplot as plt
from train import train_fn
import albumentations as A  # augmentation package
from albumentations.pytorch import ToTensorV2
from utils import *
import torch.optim as optim


def main():
    H = 80
    W = 160
    model = EfficientNetV2('b1',
                           in_channels=3,
                           in_spatial_shape=(H,W),
                           pretrained=True,
                           seg_model=True)

    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 5
    NUM_EPOCHS = 3
    NUM_WORKERS = 0
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 240
    PIN_MEMORY = False
    LOAD_MODEL = False
    # TRAIN_IMG_DIR = '/content/drive/MyDrive/cars/train/'
    # TRAIN_MASK_DIR = '/content/drive/MyDrive/cars/train_masks/'
    # VAL_IMG_DIR = '/content/drive/MyDrive/cars/val/'
    # VAL_MASK_DIR = '/content/drive/MyDrive/cars/val_masks/'
    # SAVE_DIR = "/content/drive/MyDrive/cars/images/"
    # SAVE_MODEL = "/content/drive/MyDrive/cars/UnetModels/"
    TRAIN_IMG_DIR = '/Users/Papa/cars/train/'
    TRAIN_MASK_DIR = '/Users/Papa/cars/train_masks/'
    VAL_IMG_DIR = '/Users/Papa/cars/val/'
    VAL_MASK_DIR = '/Users/Papa/cars/val_masks/'
    SAVE_DIR = "/Users/Papa/images"
    SAVE_MODEL = "effnet/"

    train_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # includes a sigmoid fn with binary cross entropy numerically more stable see docs
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms
    )

    """One common error in any large deep learning model is the problem of under flowing gradients 
    (i.e. your gradients are too small to take into account). float16 tensors often don't take into account 
    extremely small variations. To prevent this we can scale our gradients by some factor so that they aren't 
    flushed to zero. Not to be confused with vanishing gradients, this gradients still might contribute to the 
    learning process however are skipped because of computational limits."""

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    # scaler = None
    train_fn(train_loader, model, optimiser, loss_fn, scaler, NUM_EPOCHS,
             val_loader=val_loader)  # scaler val_loader=val_loader

    # Show some examples
    save_predictions_as_images(
        val_loader, model, folder=SAVE_DIR, device=DEVICE)


if __name__ == '__main__':
    # freeze_support()
    main()
