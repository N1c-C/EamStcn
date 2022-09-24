import numpy as np
import cv2 as cv
from skimage.measure import find_contours

im = np.zeros((10, 10), dtype=np.uint8)
im[2: 7, 2: 7] = 255

print(im)
# gt_mask = cv.imread('00000.png', cv.CV_8UC1)
# im = cv.imread('00000_pred.png', cv.CV_8UC1)


# im = np.zeros((10, 10), dtype=np.uint8)
# print(im, '\n\n')
idx = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
# print(idx)
out=np.zeros_like(im)
cnt=0
for x in idx:
    # print(x[0], x[0].shape)
    cnt += 1
    for arr in x:
        # out[x[0][:, 0, 0], x[0][:, 0, 1]] = 255
        print(arr[0], arr.shape)
        out[arr[0][0], arr[0][1]] = 255

    # if len(x[0].shape) > 2:

    # else:
    #     out[x[0][0], x[0][1]] = 255

# contours=find_contours(im, level=127, fully_connected='low', positive_orientation='low')
# for x in contours[0]:
#     x=np.floor(x)
#     out[int(x[1]),int (x[0])] = 255
# # print(int(np.floor(contours)))


print(out)


