from time import sleep

import torch
import albumentations as A  # augmentation package
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # progress bar
import torch.nn as nn
import torch.optim as optim
from model import UNET

from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_images

# parameters

LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 120
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'train/'
TRAIN_MASK_DIR = 'train_masks/'
VAL_IMG_DIR = 'val/'
VAL_MASK_DIR = 'val_masks/'


def train_fn(train_loader, model, optimiser, loss_fn, scaler, epochs,
             early_stop=True, val_loader=None, patience=3, save=False):

    tls = 0  # initial training loss for display on progress bar
    vls = 0 if val_loader is not None else None  # initial val loss for display on progress bar
    val_loss_hist = []  # keeps track of previous val_loss values so that early stopping can be evoked
    trn_loss_hist = []  # keep track of train loss so that it can be plotted
    stop = False  # Early stopping flag

    for epoch in range(epochs):
        if stop:
            status.write('Early stopping activated')
            break
        status = tqdm(train_loader, unit="Batch", position=0, leave=True)
        # Train Phase
        status.bar_format = "{l_bar}{bar}| Batch: {n}/{total_fmt} Time: {elapsed} < {remaining}, {rate_fmt}{postfix} "
        status.set_description(f"Epoch {epoch + 1:04}")
        status.set_postfix({"Train loss": tls})

        for batch_idx, (data, targets) in enumerate(status):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)  # squeeze to a vector of 1 along col dimension

            # forward
            model.train()
            optimiser.zero_grad()  # zero the gradients for each batch

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    preds = model(data)
                    loss = loss_fn(preds, targets)
            else:
                preds = model(data)
                loss = loss_fn(preds, targets)

            # Update Progress Bar
            status.set_postfix({"Train loss": loss.item(), "Val loss": vls})  # loss=loss.item()

            # backward
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()

            # if batch_idx > 1: # uncomment to reduce cpu commute time
            #     break

        # Validate Phase
        if val_loader is not None:
            val_loss = []
            model.eval()
            i = 0
            for inputs, targets in val_loader:
                # if i > 2:  # uncomment to reduce cpu commute time
                #     break
                inputs = inputs.to(device=DEVICE)
                targets = targets.float().unsqueeze(1).to(device=DEVICE)  # squeeze to a vector of 1 along col dimension
                val_preds = model(inputs)
                val_loss.append(loss_fn(val_preds, targets).item())
                i += 1
            vls = sum(val_loss) / len(val_loss)
            val_loss_hist.append(round(vls, 2))
            # status.clear()
            # status.set_postfix({"Train loss": tls, "Val loss": round(vls, 2)}, refresh=True)
            print(f'\rVal loss={vls:.2f}', end=' ')
            check_accuracy(val_loader, model, device=DEVICE)
            sleep(0.001)

            vls = 0

        # test for early stopping: in this case quit if loss has not reduced in the past patience number of tests
        if len(val_loss_hist) > patience and early_stop:
            past_loss = val_loss_hist[-patience:]
            stop = all(j >= i for i, j in zip(past_loss, past_loss[1:]))

        if save:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimiser': optimiser.state_dict(),
            }
            save_checkpoint(checkpoint)

    # check acc
    # check_accuracy(val_loader, model, device=DEVICE)


def main():
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

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
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

    """One common error in any large deep learning model is the problem of underflowing gradients 
    (i.e. your gradients are too small to take into account). float16 tensors often don't take into account 
    extremely small variations. To prevent this we can scale our gradients by some factor so that they aren't 
    flushed to zero. Not to be confused with vanishing gradients, this gradients still might contribute to the 
    learning process however are skipped because of computational limits."""
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    train_fn(train_loader, model, optimiser, loss_fn, scaler, NUM_EPOCHS, val_loader=val_loader)  # scaler val_loader=val_loader

    # Show some examples
    # save_predictions_as_images(
    #     val_loader, model, folder='saved_imgs/', device=DEVICE
    # )


if __name__ == '__main__':
    main()
