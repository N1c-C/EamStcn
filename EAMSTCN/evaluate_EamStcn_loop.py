"""

"""
from multiprocessing import freeze_support
from fvcore.nn import FlopCountAnalysis
import PIL.Image as Image
from torch.utils.data import DataLoader
import time
from EAMSTCN.EamStcn import EamStm
import torch
from tqdm import tqdm  # progress bar
from datasets.DavisEvalDataset import DAVISEvalDataset
# from evaluate.evaluator import EvalEamStm
from evaluate.evaluator_adaptive_save import EvalEamStm
from metrics.measurements import *
from utils import *

if __name__ == '__main__':
    freeze_support()

    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DEVICE = 'mps' if torch.backends.mps else 'cpu'

    # Data set constants
    year = 2017

    MODEL_A = '/Users/Papa/Trained Models/EamStm_A.pth'
    MODEL_B = '/Users/Papa/Trained Models/EamStm_B.pth'
    MODEL_C = '/Users/Papa/Trained Models/EamStm_C.pth'
    MODEL_Control = '/Users/Papa/Trained Models/EamStm_Control.pth'
    MODEL_Comp = '/Users/Papa/Trained Models/EamStm_Comparison.pth'

    NUMBER_OF_WORKERS = 2
    DAVIS_PATH = '/Users/Papa/'

    # Save
    name = 'b1_b1_decoder1_resFuse_pigs'
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




    # model = torch.load(MODEL_PATH)
    # model.load_state_dict(torch.load(WEIGHTS_PATH))
    models = [(MODEL_A, 'model_a'), (MODEL_B, 'model_b'), (MODEL_C, 'model_c'), (MODEL_Comp, 'comparison'),
              (MODEL_Control, 'control')]
    save_rates = [25, 10]




    # print(f'Number of parameters: {count_parameters(model)}')
    # print(f'Number of trainable parameters: {trainable_parameters(model)}')

    # Setup Datasets
    val_dataset = DAVISEvalDataset(DAVIS_PATH + '/trainval', imset=f'{year}/val.txt')
    # val_dataset = DAVISEvalDataset(DAVIS_PATH + '/test-dev', imset=f'{year}/test-dev.txt')
    val_test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUMBER_OF_WORKERS)

    results = {}
    times = {}

    for m in models:
        name = m[1]
        model = torch.load(m[0])
        model.to(torch.float32)
        model.eval().to(DEVICE)
        torch.no_grad()

        for sr in save_rates:

            RESULTS_DIR = f'/Users/Papa/Results/{name}/'
            OUTPUT_DIR = f'/content/yt_adaptive_segs/{name}/{sr}/'
            tot_fr_saved = 0
            inference = 0
            rsize = 0
            status_bar = tqdm(val_test_loader, unit="Batch", position=0, leave=True)
            for data in status_bar:
                total_frames = 0
                images = data['seq'].to(DEVICE)
                masks = (data['gt_seq'][0]).to(DEVICE)
                info = data['info']
                seq_name = info['name'][0]
                num_of_frs = images.shape[1]  # T
                height, width = images.shape[-2:]
                num_objs = len(info['labels'][0])
                size = info['size']

                status_bar.set_description(f"Sequence {seq_name}")
                evaluator = EvalEamStm(model, images, num_objs, num_of_frs, height, width, top_k=TOP_K, save_every=sr,
                                       device=DEVICE)

                # masks shape k T 1 H W  masks[:,0]  k, 1 , h, w  - 1 frame at  a time
                with torch.no_grad():  # Save some memory
                    if DEVICE == 'cuda':

                        start = time.time()
                        evaluator.evaluate(masks[:, 0], 0, num_of_frs)

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
                # ob_masks = torch.zeros(evaluator.preds.shape[0] - 1, *preds.shape)
                end = time.time()
                resize_start = time.time()
                for fr in range(evaluator.t):
                    prob = unpad(evaluator.preds[:, fr], evaluator.pad)
                    prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                    # ob_masks[:, fr] = prob[1:]
                    preds[fr] = torch.argmax(prob, dim=0)
                resize_end = time.time()

                tot_fr_saved += evaluator.model.memory.mem_size()
                inference += end - start
                rsize += resize_end - resize_start
                # ob_masks = ob_masks.detach().cpu().numpy()
                # gt = data['gts'][0].detach().cpu().numpy().astype(np.uint8)
                preds = (preds.detach().cpu().numpy()[:, 0]).astype(np.uint8)
                n_objects = evaluator.preds.shape[0]

                # Calculate the J&F for the sequence
                # results[seq_name] = calc_results(preds, gt, n_objects, start, end)

                # save images - Preds shape t, 1,w , h
                # save_images(preds, OUTPUT_DIR, seq_name, palette)

                # Save images for youtube: Every five frames and not always starting on 0, so we use the saved frame names
                # save_images_youtube(preds, OUTPUT_DIR, seq_name, palette, info)
                total_frames += len(info['frames'])
                del preds
                # del ob_masks
                # del gt
                del data
                del images
                del masks

                # purge any existing memory
                evaluator.model.memory.clear()
                del evaluator.preds

                del evaluator


            print()
            times[f'total_saved_for_{name}_sr{sr}'] = tot_fr_saved
            times[f'Inference_for_{name}_sr{sr}'] = inference
            times[f'resize_time_for_{name}_sr{sr}'] = rsize
            times[f'total_time_for_{name}_sr{sr}'] = rsize + inference

            print(f'Total frames saved for {name}_sr={sr}: ', tot_fr_saved)
            print(f'Inference time for {name}_sr={sr}: ', inference)
            print()
        del model


        # display_metrics(results, RESULTS_DIR)
    for k, v in times.items():
        print(f'{k}:   {v}')

    print(f'\nTotal frames in set: {total_frames}')