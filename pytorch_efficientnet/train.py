from multiprocessing import freeze_support

from IPython.utils.syspathcontext import prepended_to_syspath
from time import sleep
from utils import *
import torch
import albumentations as A  # augmentation package
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # progress bar
import torch.nn as nn
import torch.optim as optim

# from model import UNET

# from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_images

# parameters

LEARNING_RATE = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cpu'
BATCH_SIZE = 10
NUM_EPOCHS = 3
NUM_WORKERS = 0
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IMG_DIR = '/content/drive/MyDrive/cars/train/'
# TRAIN_MASK_DIR = '/content/drive/MyDrive/cars/train_masks/'
# VAL_IMG_DIR = '/content/drive/MyDrive/cars/val/'
# VAL_MASK_DIR = '/content/drive/MyDrive/cars/val_masks/'
# SAVE_DIR = "/content/drive/MyDrive/cars/images/"
# SAVE_MODEL = "/content/drive/MyDrive/cars/UnetModels/"
TRAIN_IMG_DIR = './vanilla_unet/train/'
TRAIN_MASK_DIR = './vanilla_unet/train_masks/'
VAL_IMG_DIR = './vanilla_unet/val/'
VAL_MASK_DIR = './vanilla_unet/val_masks/'
SAVE_DIR = "./vanilla_unet/images/"
SAVE_MODEL = "effnet/"


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
            data = data.to(DEVICE)
            targets = targets.float().unsqueeze(1).to(DEVICE)  # squeeze to a vector of 1 along col dimension

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
            status.set_postfix({"Train loss": loss.item()})  # loss=loss.item()

            # backward
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
            del preds
            del data
            del targets

            # if batch_idx > 1:
            #     break

        # Validate Phase
        if val_loader is not None:
            val_loss = []
            model.eval()
            # i = 0
            for inputs, targets in val_loader:
                # if i > 2:
                #     break
                inputs = inputs.to(device=DEVICE)
                targets = targets.float().unsqueeze(1).to(device=DEVICE)  # squeeze to a vector of 1 along col dimension
                val_preds = model(inputs)
                val_loss.append(loss_fn(val_preds, targets).item())
                # i += 1
            vls = sum(val_loss) / len(val_loss)
            val_loss_hist.append(round(vls, 2))
            # status.clear()
            # status.set_postfix({"Train loss": tls, "Val loss": round(vls, 2)}, refresh=True)
            print(f'\rVal loss={vls:.2f}', end=' ')
            check_accuracy(val_loader, model, device=DEVICE)
            sleep(0.001)
            del val_preds
            del inputs
            del targets

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
            save_checkpoint(checkpoint, SAVE_MODEL)


if __name__ == '__main__':
    freeze_support()
