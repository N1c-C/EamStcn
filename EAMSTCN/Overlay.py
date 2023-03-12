"""Simple script to overlay a predicted mask over the original image"""

import numpy as np
from PIL import Image


def overlay_davis(image, mask, colors=[255, 0, 0], cscale=2, alpha=.4):
    """ Overlay segmentation on top of RGB image - from davis official GitHub"""
    from scipy.ndimage.morphology import binary_erosion, binary_dilation
    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


if __name__ == '__main__':
    name = 'tennis_best_fr3'
    im = np.array(Image.open('/Users/Papa/test-dev/JPEGImages/480p/tennis-vest/00003.jpg'))
    mask = (Image.open('/Users/Papa/Segmentations/test_dev_fixed_sr_segs/model_a /5/tennis-vest/00003.png'))

    palette = mask.getpalette()
    mask = np.array(mask)
    colours = [[palette[i], palette[i + 1], palette[i + 2]] for i in range(0, len(palette), 3)]

    # print(palette)
    mix = overlay_davis(im, mask, colors=colours)

    # plt.imshow(mix)
    # plt.show()
    pic = Image.fromarray(mix)
    pic.save(f'/Users/Papa/Segmentations/overlays/{name}.jpg')
