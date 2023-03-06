import os
import numpy as np
import cv2 as cv
import pandas as pd
import math
from matplotlib import pyplot as plt

from EAMSTCN.metrics.ContoursAndMoments import get_contours
from skimage.morphology import disk

""" Utilities for computing, reading and saving DAVIS benchmark evaluation."""


def eval_boundary(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask.
    :param bound_th:
    :param foreground_mask: (ndarray) binary segmentation image.
    :param gt_mask:         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall

    Based on github fperazzi/davis but using openCV functions for finding contours
    """
    # tests the mask is only 2 dimensional H X W. Raise AssertionError otherwise

    print('bound', foreground_mask.shape, gt_mask.shape)
    assert len(foreground_mask.shape) == 2, "Foreground mask should be 2D (HxW)"

    # Get the pixel boundaries of both masks
    fg_boundary = get_contours(foreground_mask, bound_th=0.1)
    gt_boundary = get_contours(gt_mask)

    # Get a disk radius proportional to the size of image and
    # dilate contours proportionally
    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
    fg_dil = cv.dilate(fg_boundary, disk(bound_pix))
    gt_dil = cv.dilate(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)
    return F, precision, recall


def eval_iou(seg_mask, gt_mask):
    """ Compute region similarity (intersection over union IoU as the Jaccard Index.
    As github fperazzi/davis but with np.bool operations removed as they have been dropped.
    Mask values should be 0 or 1
    :params foreground_mask (ndarray): Predicted 2D binary annotation mask - HxW of form cv.CV_8UC1.
            gt_mask (ndarray): Provided 2D binary annotation mask - HxW of form cv.CV_8UC1.
    :returns jaccard (float): region similarity
    """
    # plt.imshow(seg_mask)
    # plt.show()

    # plt.imshow(gt_mask)
    # plt.show()
    # print(seg_mask.shape, gt_mask.shape)
    # test for a blank image  and avoid div by 0 err
    print('eval', seg_mask, gt_mask)
    if np.isclose(np.sum(gt_mask), 0) and np.isclose(np.sum(seg_mask), 0):
        return 1
    else:
        # &, |  elementwise operators
        return np.sum((gt_mask & seg_mask)) / np.sum((gt_mask | seg_mask))



def db_eval_iou(segmentation, annotation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # plt.imshow(segmentation)
    # plt.show()
    #
    # plt.imshow(annotation)
    # plt.show()

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(segmentation, annotation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def eval_tensor_iou(seg_mask, gt_mask):
    """Modified iou evaluation function for torch tensors from STM / STCN papers
    :param seg_mask :type torch.tensor()
    :param gt_mask :type torch.tensor()"""
    intersection = (gt_mask & seg_mask).float().sum()
    union = (gt_mask | seg_mask).float().sum()
    return (intersection + 1e-6) / (union + 1e-6), intersection, union


def save_results_csv(raw_results, filename):
    """Takes a dict of results for a sequence and saves them as a csv file using pandas data frames
    The df is transposed to save the metrics for sequences as rows .
    param: results: dict {'sequence': {'j': , 'f',: }}
    param: filename; full path of the file minus the extension"""
    print('Saving: ', filename + '.csv')
    pd.DataFrame(raw_results).transpose().to_csv(unique_name(filename + '.csv'))


def unique_name(filename):
    """Tests to see if a file exists and adds an incremental version number if it does.
    :param filename: full path and name for the file to be saved

    based on https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number"""

    name, ext = os.path.splitext(filename)
    cnt = 1
    newname = name
    while os.path.exists(newname + ext):
        newname = name + '_' + f'{cnt:03}'
        cnt += 1

    return newname + ext


def calc_results(preds, gt, n_objects, start, end):
    """

    :param preds:
    :param gt:
    :param n_objects:
    :param start:
    :param end:
    :return:
    """
    result = {
        'running_j': [
            sum([db_eval_iou((preds[fr] == obj), gt[fr] == obj) for obj in range(1, n_objects)])
            for fr in range(preds.shape[0])],

        'running_f': [
            sum([db_eval_boundary((preds[fr] == obj).astype(np.uint8), (gt[fr] == obj).astype(np.uint8))
                 for obj in range(1, n_objects)])
            for fr in range(preds.shape[0])],
        'num_obj': n_objects - 1,
        'time': end - start}
    return result

def display_metrics(res, save_dir=None, filename='jf_results', start_fr=1):
    """Calculates,displays and saves (optional) the average IoU(J) & Boundary values (F) and J&F for all the frames of a
    video sequence. If no file name provided the results are not saved.
     param: results: A dict of the form:
    {'seq name : {'j': [lst of J values for all frames], 'f': [lst of tuples (F, Precision, Recall) for all frames]}
    'time': time for inference}
     params: filename: Optional path/name of the file to save the results as a csv file.
     params: start_fr :int default = 1 The frame to start the calculation from. set to zero to include the ground-truth """

    calc_res = {}
    print(
        f'Sequence{" ": <15s}J{" ": <10s}F{" ": <10s}J&F{" ": <5s}Best Fr(J){" ": <5s}Worst fr(J)'
        f'{" ": <5s}Total Frs{" ": <5s}dur (secs)')

    calc_res['running_J'] = 0
    calc_res['running_F'] = 0
    calc_res['avg_J&F'] = 0
    calc_res['running_obj'] = 0
    for seq, scores in res.items():
        samples = len(scores['running_j'][start_fr:])
        f = [val for val in scores['running_f'][start_fr:]]  # unpack the tuple for F values only
        calc_res[seq] = {'J': (sum(scores["running_j"][start_fr:]) / scores['num_obj']) / samples,
                         'F': (sum(f) / scores['num_obj']) / samples,
                         'J&F': ((sum(scores["running_j"][start_fr:]) / scores['num_obj']) / samples + (sum(f) /
                                                                                    scores['num_obj']) / samples) / 2,
                         'Best Frame': np.argsort(f)[-2] + start_fr,
                         'Worst Frame': np.argmin(f) + start_fr,
                         'Total Frames': samples + start_fr,
                         'time': scores['time']}
        calc_res['running_J'] += sum(scores["running_j"][start_fr:]) / samples
        calc_res['running_F'] += sum(f) / samples
        calc_res['running_obj'] += scores['num_obj']


        print(
            f'{seq: <21s}{calc_res[seq]["J"] * 100:.2f}{"  ": <6s}',
            f'{calc_res[seq]["F"] * 100:.2f}{"  ": <7s}',
            f'{calc_res[seq]["J&F"] * 100:.2f}{"  ": <4s}',
            f'{calc_res[seq]["Best Frame"]: 5}{"  ": <11s}',
            f'{calc_res[seq]["Worst Frame"]: 5}{"  ": <11s}',
            f'{samples}{"  ": <11s}',
            f'{calc_res[seq]["time"]:.2f}')

    calc_res['avg_J'] = calc_res['running_J'] / calc_res['running_obj']
    calc_res['avg_F'] = calc_res['running_F'] / calc_res['running_obj']
    calc_res['avg_J&F'] = (calc_res['avg_J'] + calc_res['avg_F']) / 2
    print()
    print(
        f'Average{"  ": <14s}{calc_res["avg_J"] * 100:.2f}{"  ": <6s}',
        f'{calc_res["avg_F"] * 100:.2f}{"  ": <7s}',
        f'{calc_res["avg_J&F"] * 100:.2f}{"  ": <4s}'
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_results_csv(calc_res, save_dir + '/' + filename)


if __name__ == '__main__':
    # gt_mask = cv.imread('00000.png', cv.CV_8UC1)
    # foreground_mask = cv.imread('00000_pred.png', cv.CV_8UC1)
    # 
    # print(eval_boundary(foreground_mask, gt_mask, bound_th=0.008))
    # print('IoU J: ', eval_iou(foreground_mask, gt_mask))

    results = {'bear': {'f': [1, 2,3], 'j': [.8, .9,.9], 'time': 3.335665},
               'car': {'f': [6, 3,3], 'j': [.5, .3,.9], 'time': 12.5455},
               'car': {'f': [6, 3,3], 'j': [.5, .3,.9], 'time': 12.5455},
               'bat': {'f': [12, 4,3], 'j': [.8, .9,.9], 'time': 5.5455}}

    # save_results_csv(results, 'raw_results')
    # display_metrics(results, save_dir='results', filename='final_results')
    display_metrics(results)
