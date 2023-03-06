import numpy as np

from matplotlib import pyplot as plt
import PIL.Image as Image
import torch
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset


# from dataset.range_transform import im_normalization


# from dataset.util import all_to_onehot


# def all_to_onehot(masks, labels):
#     if len(masks.shape) == 3:
#         Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
#     else:
#         Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)
#
#     for k, l in enumerate(labels):
#         Ms[k] = (masks == l).astype(np.uint8)
#
#     return Ms


def to_binary(masks, labels):
    """
    Converts semantic masks with multiple objects encoded to single binary masks for each object.
    The masks are stacked in the dimension
    :param masks:  stacked np array (batch) of masks
    :param labels: A list of the labels in the masks
    :return: A batch of masks with separate channels detailing a binary mask for each object.
    """
    shape_ = list(masks.shape)
    shape_.insert(0, len(labels))  # add a dimension for each label

    bin_mask = np.zeros(shape_, dtype=np.uint8)
    for idx, obj in enumerate(labels):
        bin_mask[idx] = (masks == obj)

    return bin_mask


mask = []

h = 2
w = 2

# m = (Image.open('/Users/Papa/Jupyter_NoteBooks/Efficient_Adaptive_Memory_Stcn_Segmentation/vos_bear.png').resize(
# (w, h)).convert('P'))
m = (Image.open('//pigs.png').resize(
    (w, h)).convert('P'))
n = (Image.open('/EAMSTCN/metrics/00000.png').resize(
    (w, h)).convert('P'))
plt.imshow(m)
plt.show()
print('mask shape imported', np.shape(m))
# mask = (mask > 2)
# mask = np.where(mask > 2, 1, mask)
#
# plt.imshow(mask)
# plt.show()
num_obs = np.max(m)
mask.append(m)
# mask.append(m)

mask = np.stack(mask, 0)  # creates a batch and converts Image to numpy ndarray
print('shape passed', mask.shape)
if num_obs > 200:
    labels = [1]
    mask[0] = np.where(mask > .5, 1, mask).astype(np.uint8)
    plt.imshow(mask[0])
    plt.show()
    # print(mask[0])
    mask = torch.from_numpy(to_binary(mask, labels)).float()
    # print(mask)
else:
    labels = np.unique(mask)
    labels = labels[labels != 0]
    # print(labels)  # lose bground label
    # print(mask.shape)

    plt.imshow(mask[0])
    plt.show()
    mask = torch.from_numpy(to_binary(mask, labels)).float()

    # print(mask[0])


mask = mask.unsqueeze(2)
print(mask.shape)


# print(mask)
# for row in mask:
#     print(row)
# print(num_objects)

# colours = mask.getcolors()
# palette = mask.getpalette()  # look up putpalette
# print((palette))
# print(colours)
# Soft aggregation from STM

def aggregate(prob, keep_bg=False):
    """
    Convert a binary mask map to a softmax(logits function) aggregated map
    :param prob:
    :param keep_bg: Bool Flag: Outputs the background channel 0 or not !
    :return: tensor of
    """

    new_prob = torch.cat([
        torch.prod(1 - prob, dim=0, keepdim=True), prob], 0).clamp(1e-7, 1 - 1e-7)

    logits = torch.log((new_prob / (1 - new_prob)))

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]

# from DAVIS they load 1 frame at a time
# prob shape is ( num objects, rgb channels, 1, h, w

prob = torch.zeros((2+1, 1, 1, h, w),  dtype=torch.float32, device='cpu')
prob[0] = 1e-7
print('prob ', prob.shape)
print(prob)
# print(mask.shape)
# prob[:, 0] = aggregate(mask, keep_bg=True)
x = aggregate(mask, keep_bg=False)
print(x.shape)
prob[1:, ] = x
# print(prob)
print('prob', prob.shape)

print(prob)
