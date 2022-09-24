import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours


def get_contours_shifted(mask, bound_th=0.5):
    """ Returns one pixel wide edge contours of an image mask as a bit map/ndarray
    the contour is shifted by 1 pixel and matches the method provide for the DAVIS 16 set
    :param mask: A ground truth or predicted 2D (grayscale) image mask (H x W ),
    type CV_8UC1
    :param bound_th: The pixel threshold value for a one or zero assignment
    """

    # Tests the mask is only 2 dimensional H X W. Raise AssertionError otherwise
    assert len(mask.shape) == 2, "Foreground mask should be 2D (HxW)"

    # Convert image to true binary based on the threshold and find contours - all points
    ret, thresh = cv.threshold(mask, int(bound_th * 1), 1, 0)
    contours = find_contours(thresh, level=0.5, fully_connected='low', positive_orientation='low')

    # Create a blank canvas the same shape as the input mask and overlay the contours
    cont_matte = np.zeros_like(mask)
    for cnt in contours[0]:
        cnt = np.floor(cnt)
        cont_matte[int(cnt[0]), int(cnt[1])] = 1

    # plt.imshow(cont_matte, cmap='gray')
    # plt.show()
    return cont_matte


def get_contours(mask, bound_th=0.5):
    """ Returns one pixel wide contours of mask as a bit ndarray
    :param mask: A ground truth or predicted 2D (grayscale) image mask (H x W )
    :param bound_th: The pixel threshold value for a one or zero assignment
    """

    # Tests the mask is only 2 dimensional H X W. Raise AssertionError otherwise
    assert len(mask.shape) == 2, "Foreground mask should be 2D (HxW)"

    # Convert image to true binary based on the threshold and find contours - all points
    ret, thresh = cv.threshold(mask, int(bound_th * 1), 1, 0)
    # Set approximation as CHAIN_APPROX_NONE to keep all points (at slower speed)
    contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    # Create a blank canvas the same shape as the input mask and overlay the contours
    cont_matte = np.zeros_like(mask)
    for x in contours:
        for arr in x:
            cont_matte[arr[0][1], arr[0][0]] = 1

    # plt.imshow(cont_matte, cmap='gray')
    # plt.show()
    return cont_matte


def get_moments(cnts):
    """Returns the HU moments of a shape given a CV2 contour arary """
    return cv.moments(cnts)


def centroid(moments):
    """Returns the x and y centre positions given a cv contour array
     Calculated as Cx=M(10)/M(00) and Cy=M(01)M(00)"""
    M = moments
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    return cx, cy


def cont_area(cnts):
    """ Given a cv contour array , Returns the area bound by the contours """
    return cv.contourArea(cnts)


def disp_bound_box(img, cnts, rotated=True):
    """Returns the bounding box coordinates for a given contour array.
    Rotated option returns  a bb that has minimal area
    Otherwise the BB will span across the widest points"""
    if rotated:
        rect = cv.minAreaRect(cnts)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img, [box], 0, (0, 0, 255), 2)
    else:
        x, y, w, h = cv.boundingRect(cnts)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show()


def perimeter(cnts):
    """Given the contours of a shape returns the length of the perimeter"""
    return cv.arcLength(cnts, True)


if __name__ == '__main__':

    # load the image HxWxC for contour overlay
    img_matte = cv.imread('00000.png')
    bound_th = 0.8
    # load the image as HxW
    im = cv.imread('00000.png', cv.CV_8UC1)
    ret, thresh = cv.threshold(im, int(bound_th * 255), 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours
    cv.drawContours(img_matte, contours, -1, (0, 255, 0), 1)
    print(thresh.shape)

    cv.imshow('Contours', im)

    plt.imshow(img_matte)
    plt.show()

    cnt = contours[0]
    M = cv.moments(cnt)
    print(M)

    print(f"area = {cont_area(contours[0])}\n",
          f"Centroids = {centroid(get_moments(contours[0]))}\n",
          f"Perimeter length {perimeter(contours[0])}")
    disp_bound_box(img_matte, contours[0], rotated=True)
