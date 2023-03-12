"""
Phase one training script. Training takes place on still image sets that form sequences of three frames
The images gradually increase in size along with the augmentation transforms applied following progressive training
from EfficientNetV2: Smaller Models and Faster Training https://arxiv.org/abs/2104.00298
"""

from torch.utils.data import DataLoader, ConcatDataset
from EAMSTCN.datasets.train_datasets import *
from train.trainer import Trainer
from utils import *
import torch.optim as optim
from EAMSTCN.EamStcn import EamStcn
import torch.nn as nn

# Cuda is chosen as default if it is available
DEVICE = 'mps' if torch.backends.mps else 'cpu'
if torch.cuda.is_available():
    DEVICE = "cuda"

LEARNING_RATE = 1e-5  # The starting learning rate: Either remains constant or steps as per schedular
BATCH_SIZE = 8  # Batches of eight seem to train well
NUM_EPOCHS = 10  # The Epochs per size/gap. i.e 10 = 80 Epochs in total
NUM_WORKERS = 2  # Speed up data loading
PIN_MEMORY = True
LOG_IMG_RATE = NUM_EPOCHS / 2  # When to log images to tensorboard
SAVE_IMG_RATE = 5  # Save example predictions every x epochs
SAVE_MODEL_RATE = 2  # Save a checkpoint every x epochs

DUTS_PATH = '/Users/Papa/train/DUTS/'
PASCAL_PATH = '/Users/Papa/train/pascal/'
ECSSD_PATH = '/Users/Papa/train/ECSSD/'
HRSOD_PATH = '/Users/Papa/train/HRSOD/'

SAVE_IMG_DIR = "/Users/Papa/Results/custom_b1/"  # Location for saving example predictions
SAVE_MODEL_DIR = "/Users/Papa/Results/custom_b1/"  # Location for saving checkpoints
MODEL_NAME = 'custom1_b1_64_192_384_512_2_3_4_2_phase1'  # Name for checkpoints: the epoch number is appended to this

# Provide a path to checkpoint if restarting training OTHERWISE set as None
MODEL_PATH = '/Users/Papa/Results/custom_b1/custom1_b1_64_192_384_512_2_3_4_2_phase1_13.pth.tar'  # location of a checkpoint to load
START_FROM_EPOCH = 14  # None or Int: Set to the epoch you are starting from


def main():
    logger = Logger("")  # Starts a TensorBoard logger - Functionality limited in PyTorch

    model = EamStcn(
        key_encoder_model='custom',  # EfficientNetV2 letter, 'custom' or resnet
        value_encoder_model='b1',  # EfficientNetV2 letter or resnet
        key_pretrained=True,
        value_pretrained=True,
        in_spatial_shape=(480, 854),
        train=True,
        ck=64,
        fpn_stage=2,
        stcn_decoder=True,
        width_coefficient=None,  # Scales the width of an EfficientNetV2 key encoder
        depth_coefficient=None,  # Scales the depth of an EfficientNetV2 key encoder
        feature_expansion=False,  # Adds an extra block to the chosen EfficientNetV2 key encoder
        expansion_ch=512,  # The number of output channels for the extra block
        stage_channels=(64, 192, 384, 512),  # Custom output Ch for each stage. Only used for custom efficient encoder
        stage_depth=(2, 3, 4, 2),  # The number of repeated blocks in each stage default = 3, 4, 6, 3, 64, 256, 512,1024
        device=DEVICE
    )

    # model.load_state_dict(MODEL_PATH['state_dict'])
    #
    if MODEL_PATH is not None:
        load_checkpoint(MODEL_PATH, model)
        # load_opt_checkpoint(MODEL_PATH, optimiser)

    # Freeze BN layers

    for _, child in (model.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False

    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimiser = optim.NAdam(model.parameters(), lr=LEARNING_RATE)

    if MODEL_PATH is not None:
        optimiser.load_state_dict(torch.load(MODEL_PATH, map_location='mps')['optimiser'])

    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=25, gamma=0.1, verbose=False)

    # if MODEL_PATH is not None:
    #     scheduler.load_state_dict(torch.load(MODEL_PATH, map_location='cuda')['scheduler'])

    # scaler required when training with mixed precision to scale the gradients since they may
    # be too small to be represented by fp16
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    eam_trainer = Trainer()  # Wrapper class

    # Build datasets progressively increasing the size and the amount of augmentation. Larger images should have larger
    # transforms
    datasets = {}

    # We use square images which yield better results in general
    img_sizes = [('small', 192, 192, .4), ('mid', 224, 224, .6),
                 ('large', 336, 336, .8), ('xlarge', 384, 384, 1)]

    # lose completed datasets if they have been completed when loading a saved epoch's set of weights
    # The process sometimes can be a bit buggy with the epoch suddenly jumping a few when restarting
    if START_FROM_EPOCH is not None:
        remaining = START_FROM_EPOCH // NUM_EPOCHS
        img_sizes = img_sizes[remaining:]
        epochs_left = NUM_EPOCHS - (START_FROM_EPOCH % NUM_EPOCHS)
        so_far = START_FROM_EPOCH
    else:
        epochs_left = None
        so_far = 0

    for item in img_sizes:
        name, height, width, prog_tr_factor = item
        datasets[f'pascal_{name}'] = PascalDataSet(PASCAL_PATH, height=height, width=width, pf=prog_tr_factor)
        datasets[f'ecssd_{name}'] = EcssdDataSet(ECSSD_PATH, height=height, width=width, pf=prog_tr_factor)
        datasets[f'duts_{name}'] = DutsDataSet(DUTS_PATH, height=height, width=width, pf=prog_tr_factor)
        datasets[f'hrsod_{name}'] = HrsodDataSet(HRSOD_PATH, height=height, width=width, pf=prog_tr_factor)

    final_sets = []

    # We repeat the smaller sets to make them a bit more equal with the larger datasets
    # Maybe advantageous to just use the bigger sets
    for size in img_sizes:
        set_ = []
        pascal = [datasets['pascal_' + size[0]] for _ in range(4)]
        set_.extend(pascal)
        ecssd = [datasets['ecssd_' + size[0]] for _ in range(4)]
        set_.extend(ecssd)
        duts = [datasets['duts_' + size[0]]]
        set_.extend(duts)
        hrsod = [datasets['hrsod_' + size[0]] for _ in range(5)]
        set_.extend(hrsod)
        final_sets.append(ConcatDataset(set_))

    for data in final_sets:
        data_loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                 shuffle=True)

        eam_trainer.train(data_loader, model, optimiser, loss_fn, scaler, NUM_EPOCHS, step_lr=scheduler, so_far=so_far,
                          epochs_left=epochs_left, val_loader=None, logger=logger, device=DEVICE,
                          log_img=LOG_IMG_RATE, save_img=SAVE_IMG_RATE, model_save=SAVE_MODEL_RATE,
                          name=MODEL_NAME, SAVE_IMG_DIR=SAVE_IMG_DIR,
                          SAVE_MODEL_DIR=SAVE_MODEL_DIR)  # scaler val_loader=val_loader

        # Calculate epoch number to carry over
        if epochs_left is not None:
            so_far += (NUM_EPOCHS - epochs_left)
        else:
            so_far += NUM_EPOCHS
        epochs_left = None

    checkpoint = {
        'state_dict': model.state_dict(),
        'optimiser': optimiser.state_dict(),
    }
    save_checkpoint(checkpoint, SAVE_MODEL_DIR, f'{MODEL_NAME}_final')


if __name__ == '__main__':
    # freeze_support()
    main()
