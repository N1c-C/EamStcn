"""

"""
from torch.utils.data import DataLoader, ConcatDataset
from EAMSTCN.datasets.train_datasets import *
from train.trainer import Trainer
from utils import *
import torch.optim as optim
from EAMSTCN.EamStcn import EamStm
import torch.nn as nn

LEARNING_RATE = 1e-5
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE ='mps'
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 100
PIN_MEMORY = True
LOG_IMG_RATE = NUM_EPOCHS / 2
SAVE_IMG_RATE = 5
SAVE_MODEL_RATE = 1

DUTS_PATH = '/Users/Papa/train/DUTS/'
PASCAL_PATH = '/Users/Papa/train/pascal/'
ECSSD_PATH = '/Users/Papa/train/ECSSD/'
HRSOD_PATH = '/Users/Papa/train/HRSOD/'
# CARS_PATH = '/Users/Papa/cars/train/'

# GT_ROOT = '/Users/Papa/DAVIS_2_obj/trainval/Annotations/480p/'
# IM_ROOT = '/Users/Papa/DAVIS_2_obj/trainval/JPEGImages/480p/'
# VAL_PATH = '/Users/Papa/cars/val/'
SAVE_IMG_DIR = "/Users/Papa/Results/custom_b1/"
SAVE_MODEL_DIR = "/Users/Papa/Results/custom_b1/"
MODEL_NAME = 'custom1_b1_64_192_384_512_2_3_4_2_phase1'  # save name for checkpoints

# MODEL_PATH = '/Users/Papa/Results/test_model9.pth.tar'
MODEL_PATH = '/Users/Papa/Results/custom_b1/custom1_b1_64_192_384_512_2_3_4_2_phase1_13.pth.tar'  # location of a checkpoint to load
START_FROM_EPOCH = 14 # None or Int: Set this when loading saved weights and restarting a schedule

def main():
    logger = Logger("b3_b2_test")

    model = EamStm(
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
    # for _, child in (model.named_children()):
    #     if isinstance(child, nn.BatchNorm2d):
    #         for param in child.parameters():
    #             param.requires_grad = False

    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    # loss_fn = nn.BCEWithLogitsLoss()


    optimiser = optim.NAdam(model.parameters(), lr=LEARNING_RATE)
    if MODEL_PATH is not None:
        optimiser.load_state_dict(torch.load(MODEL_PATH, map_location='mps')['optimiser'])
    # scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=40, gamma=0.1, verbose=False)
    #
    # if MODEL_PATH is not None:
    #     scheduler.load_state_dict(torch.load(MODEL_PATH, map_location='cuda')['scheduler'])

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    eam_trainer = Trainer()  # Wrapper class

    # Build datasets progressively increasing the size and the amount of augmentation. Larger images should have larger
    # transforms
    datasets = {}

    # img_sizes = [('xsmall', 60, 112, .2), ('small', 90, 168, .4), ('mid', 120, 224, .6),
    #              ('large', 180, 336, .8), ('xlarge', 240, 448, 1)]

    img_sizes = [('small', 192, 192, .4), ('mid', 224, 224, .6),
                 ('large', 336, 336, .8), ('xlarge', 384, 384, 1)]

    # lose completed datasets if they have been completed when loading a saved epoch's set of weights
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

    # pascal_loader = DataLoader(pascal_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    #                            shuffle=True)
    # ecssd_loader = DataLoader(ecssd_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    #                           shuffle=True)
    # duts_loader = DataLoader(ecssd_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    #                          shuffle=True)
    # hrdos_loader = DataLoader(ecssd_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    #                          shuffle=True)

    # cars_data = CarsDataSet(CARS_PATH, IMAGE_HEIGHT=IMAGE_HEIGHT, IMAGE_WIDTH=IMAGE_WIDTH)
    # cars_loader = DataLoader(cars_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    #                          shuffle=True)
    # max_distance = 10
    # davis_data = VOSTrainDataset(IM_ROOT, GT_ROOT, max_distance, height=80, width=120)
    # davis_loader = DataLoader(davis_data, batch_size=16, num_workers=2, pin_memory=False, shuffle=True)
    # # val_loader = DataLoader(cars_val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    # #                         shuffle=True)

    final_sets = []

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

        eam_trainer.train(data_loader, model, optimiser, loss_fn, scaler, NUM_EPOCHS, so_far=so_far,
                          epochs_left=epochs_left, val_loader=None, logger=logger, device=DEVICE,
                          log_img=LOG_IMG_RATE, save_img=SAVE_IMG_RATE, model_save=SAVE_MODEL_RATE,
                          name=MODEL_NAME, SAVE_IMG_DIR=SAVE_IMG_DIR,
                          SAVE_MODEL_DIR=SAVE_MODEL_DIR)  # scaler val_loader=val_loader
        # either start epoch 10 , 25 or 0
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
