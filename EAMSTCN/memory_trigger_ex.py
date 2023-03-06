"""

"""
import PIL.Image as Image
from torch.utils.data import DataLoader
import time
from EAMSTCN.EamStcn import EamStm
import torch
from tqdm import tqdm  # progress bar
from datasets.DavisEvalDataset import DAVISEvalDataset
from evaluate.evaluator_adaptive_save import EvalEamStm
from metrics.measurements import *
from utils import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data set constants
# WEIGHTS_PATH = "/Users/Papa/Results/test_model5.pth.tar"
WEIGHTS_PATH = "/Users/Papa/Downloads/b3_b2_phase2_80.pth.tar"
MODEL_PATH = ''
NUMBER_OF_WORKERS = 0
DAVIS_PATH = '/Users/Papa/DAVIS_2_obj'

# Save
RESULTS_DIR = '/Users/Papa/Results/'

# Experimental Constants
TOP_K = 20
SAVE_EVERY = 5
SAVE_PR_FR = False  # save previous frame
MEMORY_SIZE = None  #
TRIGGER_VALUE = None

# palette = Image.open('/trainval/Annotations/480p/blackswan/00000.png').getpalette()
palette = Image.open('/Users/Papa/DAVIS_2_obj/00000.png').getpalette()

# Choose one or the other methods to load model

model = EamStm(
    key_encoder_model='b3',
    value_encoder_model='b2',
    key_pretrained=False,
    value_pretrained=False,
    in_spatial_shape=(480, 854),
    train=False,
    device=DEVICE
)

# model.load_state_dict(torch.load(WEIGHTS_PATH))
load_checkpoint_cuda_to_cpu(WEIGHTS_PATH, model)
# model = torch.load(MODEL_PATH)

model.eval().to(DEVICE)

# Setup Datasets
# val_dataset = DAVISEvalDataset(DAVIS_PATH + '/trainval', imset='2017/val.txt')
# val_test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUMBER_OF_WORKERS)

val_dataset = DAVISEvalDataset(DAVIS_PATH + '/trainval', imset='2017/val.txt', height=120, width=240)
val_test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUMBER_OF_WORKERS)

# test_dataset = DAVISEvalDataset(DAVIS_PATH + '/test-dev', imset='2017/test-dev.txt')
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUMBER_OF_WORKERS)

status_bar = tqdm(val_test_loader, unit="Batch", position=0, leave=True)

results = {}

for data in status_bar:

    images = data['seq'].to(DEVICE)
    masks = data['gt_seq'][0].to(DEVICE)
    info = data['info']
    seq_name = info['name'][0]
    num_of_frs = images.shape[1]  # T
    height, width = images.shape[-2:]
    num_objs = len(info['labels'][0])
    size = info['size_480p']
    status_bar.set_description(f"Sequence {seq_name}")
    evaluator = EvalEamStm(model, images, num_objs, num_of_frs, height, width, top_k=TOP_K, save_every=SAVE_EVERY,
                           memory_size=MEMORY_SIZE, save_pr_fr=SAVE_PR_FR, device=DEVICE)
    for xx in range(1, num_of_frs-1):
        # masks shape k T 1 H W  masks[:,0]  k, 1 , h, w

        with torch.no_grad():  # Save some memory
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
                with torch.cuda.amp.autocast():
                    start = time.time()
                    evaluator.evaluate(masks[:, 0], 0, num_of_frs)
                torch.cuda.synchronize()
            else:
                start = time.time()
                evaluator.evaluate(masks[:, 0], 0, num_of_frs, xx)
        end = time.time()
        preds = torch.argmax(evaluator.preds, dim=0)
        gt = torch.argmax(torch.cat([torch.zeros_like(masks), masks], dim=0), dim=0)
        preds = (preds.detach().cpu().numpy()[:, 0]).astype(np.uint8)
        gt = (gt.detach().cpu().numpy()[:, 0]).astype(np.uint8)

        # print(np.unique(preds), np.unique(data['gts']))
        # calculate
        results[seq_name] = {'j': [eval_iou((pred_mask > 0.5), (gt[idx] > 0.5)) for idx, pred_mask in enumerate(preds)],
                             'f': [eval_boundary((pred_mask > 0.5).astype(np.uint8), (gt[idx] > 0.5).astype(np.uint8)) for idx, pred_mask in enumerate(preds)],
                             'time': end-start}
        print(f'Saved frame {xx}')
        display_metrics(results, RESULTS_DIR)
        model.memory.clear()
        print()
    # results[seq_name] = {'j': [eval_iou(gt[idx], gt[idx]) for idx, pred_mask in enumerate(preds)],
    #                      'f': [eval_boundary(gt[idx], gt[idx]) for idx, pred_mask in enumerate(preds)],
    #                      'time': end-start}

    # save images

    # for fr in range(preds.shape[0]):
    #     os.makedirs(RESULTS_DIR + seq_name, exist_ok=True)
    #     img_E = Image.fromarray(preds[fr])
    #     img_E.putpalette(palette)
    #     img_E.save(os.path.join(RESULTS_DIR + seq_name, '{:05d}.png'.format(fr)))

    del preds
    del gt
    del data


display_metrics(results, RESULTS_DIR)
