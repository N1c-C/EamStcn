import os
import numpy as np
import cv2 as cv
import pandas as pd
from ContoursAndMoments import get_contours
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


def eval_iou(foreground_mask, gt_mask):

    """ Compute region similarity (intersection over union IoU as the Jaccard Index.
    As github fperazzi/davis but with np.bool operations removed as they have been dropped.
    Mask values should be 0 or 1
    :params foreground_mask (ndarray): Predicted 2D binary annotation mask - HxW of form cv.CV_8UC1.
            gt_mask (ndarray): Provided 2D binary annotation mask - HxW of form cv.CV_8UC1.
    :returns jaccard (float): region similarity

    """

    # test for a blank image  and avoid div by 0 err
    if np.isclose(np.sum(gt_mask), 0) and np.isclose(np.sum(foreground_mask), 0):
        return 1
    else:
        # &, |  elementwise operators
        return np.sum((gt_mask & foreground_mask)) / np.sum((gt_mask | foreground_mask))


def save_results_csv(raw_results, filename):
    """Takes a dict of results for a sequence and saves them as a csv file using pandas data frames
    The df is transposed to save the metrics for sequences as rows .
    param: results: dict {'sequence': {'j': , 'f',: }}
    param: filename; full path of the file minus the extension"""
    pd.DataFrame(raw_results).transpose().to_csv(unique_name(filename +'.csv'))


def unique_name(filename):
    """Tests to see if a file exists and adds an incremental version number if it does.
    :param filename: full path and name for the file to be saved

    based on https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number"""

    name, ext = os.path.splitext(filename)
    cnt = 1
    newname = name
    while os.path.exists(newname+ext):
        newname = name + '_' + f'{cnt:03}'
        cnt += 1

    return newname+ext


def display_metrics(res, filename=None):
    """Calculates,displays and saves (optional) the average IoU(J) & Boundary values (F) and J&F for all the frames of a
    video sequence. If no file name provided the results are not saved

     param: results: A dict of the form:
    {'seq name : {'j': [lst of J values for all frames], 'f': [lst of tuples (F, Precision, Recall) for all frames]}}
     params: filename: Optional path/name of the file to save the results as a csv file."""

    calc_res = {}
    print(
        f'Sequence{" ": <15s}J{" ": <10s}F{" ": <10s}J&F{" ": <8s}Best Frame(J){" ": <8s}Worst frame(J)'
        f'{" ": <8s}Total Frames')

    for seq, scores in res.items():
        samples = len(scores['j'])
        f = [val[0] for val in scores['f']]  # unpack the tuple for F values only
        calc_res[seq] = {'J': sum(scores["j"]) / samples,
                         'F': sum(f) / samples,
                         'J&F': (sum(scores["j"]) / samples + sum(f) / samples) / 2,
                         'Best Frame': np.argmax(f),
                         'Worst Frame': np.argmin(f),
                         'Total Frames': samples}
        print(
            f'{seq: <18s}    {calc_res[seq]["J"]:.2f}',
            f'      {calc_res[seq]["F"]:.2f}',
            f'      {calc_res[seq]["J&F"]:.2f}',
            f'          {calc_res[seq]["Best Frame"]: 5}{"  ": <6s}',
            f'          {calc_res[seq]["Worst Frame"]: 5}{"  ": <6s}'
            f'          {samples}')
    if filename is not None:
        save_results_csv(calc_res, filename)


if __name__ == '__main__':
    
    # gt_mask = cv.imread('00000.png', cv.CV_8UC1)
    # foreground_mask = cv.imread('00000_pred.png', cv.CV_8UC1)
    # 
    # print(eval_boundary(foreground_mask, gt_mask, bound_th=0.008))
    # print('IoU J: ', eval_iou(foreground_mask, gt_mask))

    results = {'bear': {'f':[(1,2,3),(2,3,4)], 'j':[.8,.9]},'car': {'f':[(6,7,1),(3,3,8)], 'j':[.5,.3]}}

    # save_results_csv(results, 'raw_results')
    display_metrics(results, 'final_results')
    display_metrics(results)


