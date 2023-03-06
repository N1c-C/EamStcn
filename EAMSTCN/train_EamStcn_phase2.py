"""

"""
from multiprocessing import freeze_support

from torch.utils.data import DataLoader, ConcatDataset
from EAMSTCN.datasets.train_datasets import *
from train.trainer import Trainer
from utils import *
import torch.optim as optim
from EAMSTCN.EamStcn import EamStm
import torch.nn as nn

LEARNING_RATE = 1e-5
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'mps'
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 4
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 100
PIN_MEMORY = True
LOG_IMG_RATE = NUM_EPOCHS / 2
SAVE_IMG_RATE = 5
SAVE_MODEL_RATE = 2

# GT_ROOT = '/Users/Papa/DAVIS_2_obj/trainval/Annotations/480p/'
# IM_ROOT = '/Users/Papa/DAVIS_2_obj/trainval/JPEGImages/480p/'

ROOT = '/Users/Papa/trainval/'
YTV_ROOT = '/Users/Papa/yv_train'

SAVE_IMG_DIR = "/Users/Papa/Results/b1_b1_ex512/images/"
SAVE_MODEL_DIR = "/Users/Papa/Results/b1_b1_ex512/checkpoints/"
MODEL_NAME = 'b1_b1_ex1024_stcn_512_512_decoder_square_yt19'  # save name for checkpoints

MODEL_PATH = '/Users/Papa/Trained Models/b1_b1/b1_b1_stcn_dec_ck64_ex512_phase2_yt19_no_amp_46.pth.tar'
# MODEL_PATH = None  # location of a checkpoint to load
START_FROM_EPOCH = None  # None or Int: Set this when loading saved weights and restarting a schedule


def main():
    logger = Logger("b3_b2_test")

    model = EamStm(
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

    # # Change ck to 128 so remove proj key parameters
    if MODEL_PATH is not None:
        sd = (torch.load(MODEL_PATH, map_location=torch.device('mps'))['state_dict'])
        x = sd.copy()
        for key, value in x.items():


            # if  in key:
            #     print(key)
            #     sd.pop(key)
            if 'compress' or 'feature_expansion' or 'key_projection' in key:
                sd.pop(key)


        model.load_state_dict(sd, strict=False)

    # Freeze BN layers
    for _, child in (model.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False

    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    # # loss_fn = nn.BCEWithLogitsLoss()
    optimiser = optim.NAdam(model.parameters(), lr=LEARNING_RATE)
    # if MODEL_PATH is not None:
    #     optimiser.load_state_dict(torch.load(MODEL_PATH, map_location='mps')['optimiser'])

    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=40, gamma=0.2, verbose=False)
    # scheduler = None
    # if MODEL_PATH is not None:
    #     scheduler.load_state_dict(torch.load(MODEL_PATH, map_location='mps')['schedular'])
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    eam_trainer = Trainer()  # Wrapper class for training function

    # max_distance = 10
    # davis_data = VOSTrainDataset(IM_ROOT, GT_ROOT, max_distance, height=80, width=120)
    # davis_loader = DataLoader(davis_data, batch_size=16, num_workers=2, pin_memory=False, shuffle=True)
    # davis_data = VOSTrainDataset(IM_ROOT, GT_ROOT, max_distance, height=80, width=120)
    # davis_loader = DataLoader(davis_data, batch_size=16, num_workers=2, pin_memory=False, shuffle=True)
    # val_loader = DataLoader(cars_val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    #                         shuffle=True)

    # Build datasets progressively increasing the size and the amount of augmentation. Larger images should have larger
    # transforms
    datasets = {}

    vos_sizes = [('small', 192, 192, .6, 10), ('mid', 224, 224, .8, 15),
                 ('large', 256, 256, 1, 20), ('xlarge', 384, 384, 1.2, 25), ('taper1', 384, 384, 1, 15),
                 ('taper2', 384, 384, .8, 10), ('taper3', 384, 384, .6, 7), ('taper4', 384, 384, .4, 5)]

    # vos_sizes = [('small', 112, 176, .6, 10), ('mid', 128, 224, .8, 15),
    #               ('large', 196, 336, 1, 20), ('xlarge', 256, 448, 1.2, 25), ('taper1', 256, 448, 1, 20),
    #               ('taper2', 256, 448, .8, 15), ('taper3', 256, 448, .6, 10), ('taper4', 256, 448, .4, 5)]

    # vos_sizes = [('xsmall', 60, 112, .4, 5), ('small', 90, 168, .6, 10), ('mid', 120, 224, .8, 15),
    #              ('large', 180, 336, 1, 20), ('xlarge', 240, 448, 1.2, 25), ('taper1', 240, 448, 1, 20),
    #              ('taper2', 240, 448, .8, 15), ('taper3', 240, 448, .6, 10), ('taper4', 240, 448, .4, 5)]

    # vos_sizes = [('taper1', 384, 384, 1, 20),
    # #              ('taper2', 384, 384, .8, 15), ('taper3', 240, 432, .6, 10), ('taper4', 240, 432, .4, 5)]
    # vos_sizes = [
    #     ('small', 192, 192, .4, 10), ('mid', 256, 256, .6, 15),
    #     ('large', 320, 320, .8, 20), ('xlarge', 384, 384, 1, 25), ('taper1', 384, 384, 1, 20),
    #     ('taper2', 384, 384, 1, 15), ('taper3', 384, 384, 1, 10), ('taper4', 384, 384, 1, 5)]

    # vos_sizes = [
    #              ('large', 180, 336, .8, 20), ('xlarge', 240, 448, 1.4, 25), ('taper1', 240, 448, 1.2, 20),
    #              ('taper2', 240, 448, .8, 15), ('taper3', 240, 448, .8, 10), ('taper4', 240, 448, .4, 5)]

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

        # scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1, verbose=True)
        data_loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                 shuffle=True)

        eam_trainer.train(data_loader, model, optimiser, loss_fn, scaler, NUM_EPOCHS, step_lr=scheduler,
                          so_far=so_far, epochs_left=epochs_left,
                          val_loader=None, logger=logger, device=DEVICE,
                          log_img=LOG_IMG_RATE, save_img=SAVE_IMG_RATE, model_save=SAVE_MODEL_RATE,
                          name=MODEL_NAME, SAVE_IMG_DIR=SAVE_IMG_DIR,
                          SAVE_MODEL_DIR=SAVE_MODEL_DIR)  # scaler val_loader=val_loader
        # either start epoch 10 , 25 or 0
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
