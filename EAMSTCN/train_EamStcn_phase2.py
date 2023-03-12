"""
Phase two training script. Training takes place on three temporally ordered frames from the DAVIS and YouTube datasets
The images gradually increase in size along with the augmentation transforms applied following progressive training
from EfficientNetV2: Smaller Models and Faster Training https://arxiv.org/abs/2104.00298
The gap between the frames increases from 5...to 25....and back to 5 as per https://github.com/seoungwugoh/STM
"""
from multiprocessing import freeze_support
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

# The locations for the datasets to be used
ROOT = '/Users/Papa/trainval/'
YTV_ROOT = '/Users/Papa/yv_train'

SAVE_IMG_DIR = "/Users/Papa/Results/b1_b1_ex512/images/"  # Location for saving example predictions
SAVE_MODEL_DIR = "/Users/Papa/Results/b1_b1_ex512/checkpoints/"  # Location for saving checkpoints
MODEL_NAME = 'b1_b1_ex1024_stcn_512_512_decoder_square_yt19'  # Name for checkpoints: epoch number is appended to this

# Provide a path to checkpoint if restarting training OTHERWISE set as None
MODEL_PATH = '/Users/Papa/Trained Models/b1_b1/b1_b1_stcn_dec_ck64_ex512_phase2_yt19_no_amp_46.pth.tar'
START_FROM_EPOCH = None  # None or Int: Set this when loading saved weights and restarting a schedule


def main():
    logger = Logger("b3_b2_test")  # Starts a TensorBoard logger - Functionality limited in PyTorch

    model = EamStcn(
        key_encoder_model='b1',  # EfficientNetV2 letter or resnet
        value_encoder_model='b1',  # EfficientNetV2 letter or resnet
        key_pretrained=False,
        value_pretrained=False,
        in_spatial_shape=(480, 854),
        train=True,
        ck=64,
        fpn_stage=2,
        stcn_decoder=True,
        width_coefficient=None,  # Scales the width of an EfficientNetV2 key encoder
        depth_coefficient=None,  # Scales the depth of an EfficientNetV2 key encoder
        feature_expansion=True,  # Adds an extra block to the chosen EfficientNetV2 key encoder
        expansion_ch=1024,  # The number of output channels for the extra block
        stage_channels=(64, 192, 384, 768),  # Custom output Ch for each stage. Only used for custom efficient encoder
        stage_depth=(2, 3, 5, 2),  # The number of repeated blocks in each stage default = 3, 4, 6, 3, 64, 256, 512,1024
        device=DEVICE
    )

    # if MODEL_PATH is not None:
    #     # load_checkpoint(MODEL_PATH, model)
    #     # load_opt_checkpoint(MODEL_PATH, optimiser)
    #     sd = (torch.load(MODEL_PATH, map_location=torch.device('mps'))['state_dict'])
    #     model.load_state_dict(sd, strict=False)

    """When transfer learning - we can pop the weighs from the dictionary that are no longer relevant before 
    loading the rest"""

    # if MODEL_PATH is not None:
    #     sd = (torch.load(MODEL_PATH, map_location=torch.device('mps'))['state_dict'])
    #     x = sd.copy()
    #
    #     for key, value in x.items():
    #         if 'compress' or 'feature_expansion' or 'key_projection' in key:
    #             sd.pop(key)
    #     model.load_state_dict(sd, strict=False)

    # Freeze BN layers
    for _, child in (model.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False

    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    optimiser = optim.NAdam(model.parameters(), lr=LEARNING_RATE)
    # if MODEL_PATH is not None:
    #     optimiser.load_state_dict(torch.load(MODEL_PATH, map_location='mps')['optimiser'])

    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=25, gamma=0.1, verbose=False)
    # scheduler = None
    # if MODEL_PATH is not None:
    #     scheduler.load_state_dict(torch.load(MODEL_PATH, map_location='mps')['schedular'])

    # scaler required when training with mixed precision to scale the gradients since they may
    # be too small to be represented by fp16
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    eamstcn_trainer = Trainer()  # Wrapper class for training function

    # Build datasets progressively increasing the size and the amount of augmentation. Larger images should have larger
    # transforms
    datasets = {}

    vos_sizes = [('small', 192, 192, .6, 10), ('mid', 224, 224, .8, 15),
                 ('large', 256, 256, 1, 20), ('xlarge', 384, 384, 1.2, 25), ('taper1', 384, 384, 1, 15),
                 ('taper2', 384, 384, .8, 10), ('taper3', 384, 384, .6, 7), ('taper4', 384, 384, .4, 5)]

    # lose completed datasets when loading a saved epoch's set of weights
    if START_FROM_EPOCH is not None:
        remaining = START_FROM_EPOCH // NUM_EPOCHS
        vos_sizes = vos_sizes[remaining:]
        epochs_left = NUM_EPOCHS - (START_FROM_EPOCH % NUM_EPOCHS)
        so_far = START_FROM_EPOCH
    else:
        epochs_left = None
        so_far = 0

    for item in vos_sizes:
        name, height, width, prog_tr_factor, max_distance = item
        datasets[f'davis_{name}'] = VOSTrainDataset(ROOT, imset='2017', max_jump=max_distance,
                                                    height=height, width=width, pf=prog_tr_factor)

        datasets[f'youtube_{name}'] = VOSTrainDataset(YTV_ROOT, youtube=True, max_jump=max_distance // 5,
                                                      height=height, width=width, pf=prog_tr_factor)

    final_sets = []

    # double sets as the size increases but single ones as we taper down the distance between frames
    for size in vos_sizes[:-4]:
        ds = [datasets['davis_' + size[0]] for _ in range(75)]
        ytds = [datasets['youtube_' + size[0]] for _ in range(3)]
        final_sets.append(ConcatDataset(ds + ytds))

    for size in vos_sizes[-4:]:
        ds = [datasets['davis_' + size[0]] for _ in range(75)]
        ytds = [datasets['youtube_' + size[0]] for _ in range(3)]
        final_sets.append(ConcatDataset(ds + ytds))

    for data in final_sets:

        data_loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                 shuffle=True)

        eamstcn_trainer.train(data_loader, model, optimiser, loss_fn, scaler, NUM_EPOCHS, step_lr=scheduler,
                              so_far=so_far, epochs_left=epochs_left,
                              val_loader=None, logger=logger, device=DEVICE,
                              log_img=LOG_IMG_RATE, save_img=SAVE_IMG_RATE, model_save=SAVE_MODEL_RATE,
                              name=MODEL_NAME, SAVE_IMG_DIR=SAVE_IMG_DIR,
                              SAVE_MODEL_DIR=SAVE_MODEL_DIR)

        # Calculate epoch number to carry over
        if epochs_left is not None:
            so_far += epochs_left
        else:
            so_far += NUM_EPOCHS
        epochs_left = None

    checkpoint = {
        'state_dict': model.state_dict(),
        'optimiser': optimiser.state_dict(),
    }
    save_checkpoint(checkpoint, SAVE_MODEL_DIR, f'{MODEL_NAME}_final')


if __name__ == '__main__':
    freeze_support()
    main()
