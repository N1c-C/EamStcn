"""

"""
from multiprocessing import freeze_support
from fvcore.nn import FlopCountAnalysis
import PIL.Image as Image
from torch.utils.data import DataLoader
import time
from EAMSTCN.EamStcn import EamStcn
import torch
from tqdm import tqdm  # progress bar
from datasets.DavisEvalDataset import DAVISEvalDataset
from evaluate.evaluator import EvalEamStm
from metrics.measurements import *
from utils import *

if __name__ == '__main__':
    freeze_support()

    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DEVICE = 'mps' if torch.backends.mps else 'cpu'

    # Data set constants
    year = 2017

    # MODEL_A = '/Users/Papa/Trained Models/EamStm_A.pth'
    # MODEL_PATH = '/Users/Papa/Trained Models/EamStm_B.pth'
    MODEL_PATH = '/Users/Papa/Trained Models/EamStm_C.pth'
    # MODEL_PATH = '/Users/Papa/Trained Models/EamStm_Control.pth'
    # MODEL_PATH = '/Users/Papa/Trained Models/EamStm_Comparison.pth'

    NUMBER_OF_WORKERS = 2
    DAVIS_PATH = '/Users/Papa/'

    # Save
    name = 'model_C_val'
    RESULTS_DIR = f'/Users/Papa/Results/{name}/'
    OUTPUT_DIR = f'/Users/Papa/Segmentations/{name}/'

    # Experimental Constants
    TOP_K = 16
    SAVE_EVERY = 5
    SAVE_PR_FR = False  # save previous frame
    MEMORY_SIZE = None  #
    TRIGGER_VALUE = None

    # palette = Image.open('/trainval/Annotations/480p/blackswan/00000.png').getpalette()
    palette = Image.open('/Users/Papa/DAVIS_2_obj/00000.png').getpalette()

    # Choose one or the other methods to load model
    WEIGHTS_PATH = "/Users/Papa/Trained Models/b3_b2/b3_b2_fpn2_ck64_phase2_youtube_no_amp_start_from_p1_96_plus_final.pth.tar"
    # WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1_ex512/b1_b1_ex512_MixedFuse_stcn512_256_dec_phase2_yt19_amp_final.pth.tar'  # 82.76
    WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1_ex512/b1_b1_stcn_512_256_ex512_phase2_yt19amp_bse_60.pth.tar'
    # WEIGHTS_PATH = '/Users/Papa/Trained Models/b3_b2/b3_b2_stcn512_256_dec_phase2_yt18_amp_final.pth.tar'
    WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1_ex512/b1_b1_stcn_dec_ck64_ex512_phase2_yt19_no_amp_64.pth.tar' # 80.48
    # WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1/b1_b1_stcn512_256_ck64_phase2_yt19_amp_66.pth.tar'
    WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1/b1_b1_stcn512_256_ck64_phase2_yt19_resFuseamp_final.pth.tar' # 80.43
    # model = EamStcn(
    #     key_encoder_model='b1',  # EfficientNetV2 letter, 'resnet' or 'custom'
    #     value_encoder_model='b1',  # EfficientNetV2 letter, 'resnet'
    #     key_pretrained=False,
    #     value_pretrained=False,
    #     in_spatial_shape=(480, 854),
    #     train=False,
    #     ck=64,
    #     fpn_stage=2,
    #     decoder='fpn',
    #     width_coefficient=None,  # Scales the width of an EfficientNetV2 key encoder
    #     depth_coefficient=None,  # Scales the depth of an EfficientNetV2 key encoder
    #     feature_expansion=True,  # Adds an extra block to the chosen EfficientNetV2 key encoder
    #     expansion_ch=512,  # The number of output channels for the extra block
    #     stage_channels=(64, 192, 384, 512),  # Custom output Ch for each stage. Only used for custom efficient encoder
    #     stage_depth=(2, 3, 4, 2),  # The number of repeated blocks in each stage default = 3, 4, 6, 3, 64, 256, 512,1024
    #     device=DEVICE
    # )

    # load_checkpoint_cuda_to_device(WEIGHTS_PATH, model, DEVICE)
    # print("Alert NOTHING LOADED")

    # sd = torch.load(WEIGHTS_PATH, map_location=torch.device('mps'))['state_dict']
    # x = sd.copy()
    # for key, value in x.items():
    #     if 'decoder' in key:
    #         sd.pop(key)
    #
    # model.load_state_dict(sd, strict=False)


    model = torch.load(MODEL_PATH)
    # model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval().to(DEVICE)


    print(f'Number of parameters: {count_parameters(model)}')
    print(f'Number of trainable parameters: {trainable_parameters(model)}')

    # Setup Datasets
    val_dataset = DAVISEvalDataset(DAVIS_PATH + '/trainval', imset=f'{year}/test.txt')
    # val_dataset = DAVISEvalDataset(DAVIS_PATH + '/test-dev', imset=f'{year}/test-dev.txt')
    val_test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUMBER_OF_WORKERS)

    status_bar = tqdm(val_test_loader, unit="Batch", position=0, leave=True)

    results = {}
    tot_fr_saved = 0

    for data in status_bar:

        images = data['seq'].to(DEVICE)
        masks = (data['gt_seq'][0]).to(DEVICE)
        info = data['info']
        seq_name = info['name'][0]
        num_of_frs = images.shape[1]  # T
        height, width = images.shape[-2:]
        num_objs = len(info['labels'][0])
        size = info['size']
        status_bar.set_description(f"Sequence {seq_name}")
        evaluator = EvalEamStm(model, images, num_objs, num_of_frs, height, width, top_k=TOP_K, save_every=SAVE_EVERY,
                               memory_size=MEMORY_SIZE, save_pr_fr=SAVE_PR_FR, device=DEVICE)

        # masks shape k T 1 H W  masks[:,0]  k, 1 , h, w  - 1 frame at  a time

        with torch.no_grad():  # Save some memory
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
                with torch.cuda.amp.autocast():
                    start = time.time()
                    evaluator.evaluate(masks[:, 0], 0, num_of_frs)
                torch.cuda.synchronize()
            elif DEVICE == 'mps':
                start = time.time()
                evaluator.evaluate(masks[:, 0], 0, num_of_frs)
            else:
                with torch.autocast('cpu'):
                    start = time.time()
                    evaluator.evaluate(masks[:, 0], 0, num_of_frs)

        # Unpad images and resample to the correct shape
        preds = torch.zeros((evaluator.t, 1, *size), dtype=torch.uint8, device=DEVICE)

        # ob_masks used for immediate scoring - shape: n_objs-1 (background) , t, 1, h, w
        ob_masks = torch.zeros(evaluator.preds.shape[0] - 1, *preds.shape)
        end = time.time()

        # Resize the masks
        for fr in range(evaluator.t):
            prob = unpad(evaluator.preds[:, fr], evaluator.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            ob_masks[:, fr] = prob[1:]
            preds[fr] = torch.argmax(prob, dim=0)

        tot_fr_saved += evaluator.model.memory.mem_size()

        ob_masks = ob_masks.detach().cpu().numpy()
        gt = data['gts'][0].detach().cpu().numpy().astype(np.uint8)
        preds = (preds.detach().cpu().numpy()[:, 0]).astype(np.uint8)
        n_objects = evaluator.preds.shape[0]

        # Calculate the J&F for the sequence
        results[seq_name] = calc_results(preds, gt, n_objects, start, end)

        # save images - Preds shape t, 1,w , h
        # save_images(preds, OUTPUT_DIR, seq_name, palette)

        # Save images for youtube: Every five frames and not always starting on 0, so we use the saved frame names
        save_images_youtube(preds, OUTPUT_DIR, seq_name, palette, info)

        del preds
        del ob_masks
        del gt
        del data

    print()
    display_metrics(results, RESULTS_DIR, filename=name + f'_DAVIS-{year}')
    print()
    print('Total Frames Saved: ', tot_fr_saved)


    # img = Image.open('/Users/Papa/trainval/JPEGImages/480p/bike-packing/00068.jpg')
# ob_masks = ob_masks.detach().cpu().numpy()
# gt = (masks.detach().cpu().numpy()).astype(np.uint8)

# k, t, c, h, w = preds.shape
# save_im = np.zeros((k + 1, h, w))
# for obj in range(preds.shape[0]):
#     save_im[obj + 1][(preds[obj][fr][0] > 0.43)] = obj + 1
# save_im = np.argmax(save_im, axis=0).astype(np.uint8)


    # imx = torch.rand(1, 3, 480, 854).to(DEVICE)
    # imy = torch.rand(1, 5, 480, 854).to(DEVICE)
    # y = model.value_encoder
    # x = model.key_encoder
    # z = model.decoder
    # flopx = FlopCountAnalysis(x, imx)
    # flopy = FlopCountAnalysis(y, imy)
    # flopz = FlopCountAnalysis(z, model.q)
    #
    # print(f'Flop Count: {(flopx.total() + flopy.total()) /1e9:.3f} GFlops')

# results[seq_name] = {
#     'j': [
#         sum([eval_iou((ob_masks[obj][fr][0] > 0.5), gt[obj][fr][0]) for obj in range(ob_masks.shape[0])]) /
#         ob_masks.shape[0]
#         for fr in range(ob_masks.shape[1])],
#
#     'f': [
#         sum([eval_boundary((ob_masks[obj][fr][0] > 0.5).astype(np.uint8),
#                            (gt[obj][fr][0] > 0.5).astype(np.uint8))[0]
#              for obj in range(ob_masks.shape[0])]) / ob_masks.shape[0]
#         for fr in range(ob_masks.shape[1])],
#     'time': end - start}

# # Score from argmax
#
#

# gt = torch.argmax(torch.cat([torch.zeros_like(masks), masks], dim=0), dim=0)
# # gt = torch.argmax(masks, dim=0)

#
# results[seq_name] = {'j': [eval_iou(pred_mask, gt[idx]) for idx, pred_mask in enumerate(preds)],
#                      'f': [eval_boundary(pred_mask, gt[idx])[0] for idx, pred_mask in enumerate(preds)],
#                      'time': end - start}
