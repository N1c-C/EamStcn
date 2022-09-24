import numpy as np
import cv2 as cv
from contours_moments import get_contours
from skimage.morphology import disk

""" Utilities for computing, reading and saving DAVIS benchmark evaluation."""


def eval_boundary(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask.
    :param foreground_mask: (ndarray) binary segmentation image.
    :param gt_mask:         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall

    Based on github fperazzi/davis but using openCV functions for finding contours
    and dilation to significantly improve speed
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


if __name__ == '__main__':
    gt_mask = cv.imread('00000.png', cv.CV_8UC1)
    foreground_mask = cv.imread('00000_pred.png', cv.CV_8UC1)

    print(eval_boundary(foreground_mask, gt_mask, bound_th=0.008))
    print('IoU J: ', eval_iou(foreground_mask, gt_mask))
