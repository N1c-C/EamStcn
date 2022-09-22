import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# load the image HxWxC for contour overlay
img_matte = cv.imread('00000.png')

# load the image as HxW
im = cv.imread('00000.png', cv.CV_8UC1)

contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Draw contours
cv.drawContours(img_matte, contours, -1, (0, 255, 0), 1)
print(im.shape)

cv.imshow('Contours', im)

plt.imshow(img_matte)
plt.show()

cnt = contours[0]
M = cv.moments(cnt)
print(M)


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
    return cv.contourArea(cnt)


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


print(f"area = {cont_area(contours[0])}\n",
      f"Centroids = {centroid(get_moments(contours[0]))}\n",
      f"Perimeter length {perimeter(contours[0])}")
disp_bound_box(img_matte, contours[0], rotated=True)
