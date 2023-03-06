import torch
import torchvision
import numpy as np
import PIL.Image as Image
from albumentations.augmentations import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os


def save_checkpoint(state, path, filename='test_checkpoint'):
    os.makedirs(path, exist_ok=True)
    filename = filename + '.pth.tar'
    path = path + filename
    print('=> Saving checkpoint: ', filename)
    torch.save(state, path)


def load_checkpoint(path, model):
    print('=> Loading checkpoint: ', path)
    model.load_state_dict(torch.load(path)['state_dict'])


def load_checkpoint_cuda_to_device(path, model, device):
    print('=> Loading checkpoint: ', path)
    model.load_state_dict(torch.load(path, map_location=torch.device(device))['state_dict'])


def load_opt_checkpoint(path, opt):
    print('=> Loading optimiser: ', path)
    opt.load_state_dict(torch.load(path)['optimiser'])


def load_model(path):
    print('=> Loading model: ', path)
    return torch.load(path)


def save_model(path, model):
    print('=> Loading model: ', path)
    torch.save(model, path)


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    """Disabling gradient calculation is useful for inference, 
    when you are sure that you will not call Tensor.backward(). 
    It will reduce memory consumption for computations that would otherwise have requires_grad=True."""

    with torch.no_grad():
        for x, y, in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)  # Returns the total number of elements in the input tensor
            dice_score += (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8

    print(f'Accuracy: {(num_correct / num_pixels) * 100:.2f}%, Dice Score: {dice_score / len(loader):.3f}')
    model.train()


def save_images(preds, output_dir, seq_name, palette):
    """

    :param preds:
    :param output_dir:
    :param seq_name:
    :param palette:
    :return:
    """
    for fr in range(preds.shape[0]):
        os.makedirs(output_dir + seq_name, exist_ok=True)
        img = Image.fromarray(preds[fr])
        img.putpalette(palette)
        img.save(os.path.join(output_dir + seq_name, f'{fr:05d}.png'))


def save_images_youtube(preds, output_dir, seq_name, palette, info):
    """
    :param preds:
    :param output_dir:
    :param seq_name:
    :param palette:
    :param info:
    :return:
    """
    for fr in range(preds.shape[0]):
        os.makedirs(output_dir + seq_name, exist_ok=True)
        img = Image.fromarray(preds[fr])
        img.putpalette(palette)
        name = info['frames'][fr][0].replace('.jpg', '.png')
        img.save(os.path.join(output_dir + seq_name, f'{name}'))



def one_hot_mask(masks, dim=0, start_label=1):
    """Converts semantic masks with multiple objects encoded as an int (0,1,2,etc) to single binary masks
    for each object.
    :param dim: the dimension to stack the masks in
    :param masks:  batch of masks
    :param start_label:
    :return: tuple: A batch of masks with separate channels detailing a binary mask for each object
    and tensor of the labels
    """
    labels = torch.unique(masks)[start_label:]
    return torch.stack([masks == obj for obj in labels], dim=dim).float(), labels


def aggregate(prob, dim=0, keep_bg=True):
    """ STM multi object aggregation function - see paper's supplementary section
     :returns the probability of the logits function either with or without the background as a mask prediction"""

    new_prob = torch.cat([
        torch.prod(1 - prob, dim=dim, keepdim=True), prob
    ], dim).clamp(1e-7, 1 - 1e-7)
    logits = torch.log((new_prob / (1 - new_prob)))
    if keep_bg:
        return logits
    else:
        return logits[1:]


def trainable_parameters(model):
    """
    from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
    :param model: A PyTorch model
    :return: The number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    """
    from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
    :param model: A PyTorch model
    :return: The number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters())

def get_other_objects_mask(mask):
    """
    Modified from STM. Returns a mask of the other objets - per mask channel.
    i.e channel one will contain bits representing all other objects except object one.
    Passed with the mask to the encoder to aid discrimination between objects.
    :param mask: Tensor shape: objs, 1, H, W - obj is one dim/object
    :return: Tensor Num_objs, 1, H, W
    """
    num_objs = mask.shape[0]
    if num_objs != 1:
        other_objs = torch.cat([
            torch.sum(mask[[obj for obj in range(num_objs) if i != obj]], dim=0, keepdim=True)
            for i in range(num_objs)], 0)

    else:
        other_objs = torch.zeros_like(mask)
    return other_objs


class Logger:
    """A function to log to tensorboard to be extended as required
    If no log path is provided then the tensorboard default /run directory is used"""

    def __init__(self, log_dir=None):
        self.logger = SummaryWriter() if log_dir is not None else SummaryWriter(log_dir)
        # self.inv_im_trans = transforms.Normalize(
        #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    def add_scalar(self, title, value, epoch):
        self.logger.add_scalar(title, value, epoch)

    def add_metrics(self):
        pass

    def add_images(self, title, image, epoch):
        """Param: title:str: a valid name/path for the images
        :param: image: torch tensor
        :param: epoch: int: """
        im = torchvision.utils.make_grid(image, normalize=True).unsqueeze(0)
        self.logger.add_images(title, im, epoch)

    def add_masks(self, title, mask, epoch):
        """Param: title:str: a valid name/path for the images
        :param: mask: torch tensor
        :param: epoch: int: """
        self.logger.add_images(title, mask, epoch)

    def flush(self):
        self.logger.flush()

    def close(self):
        self.logger.close()


# STM
def pad_divide_by(in_img, d, in_size=None):
    if in_size is None:
        h, w = in_img.shape[-2:]
    else:
        h, w = in_size

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    """The padding size by which to pad dimensions of input are described starting from the last
    dimension and moving forward"""
    out = F.pad(in_img, pad_array)
    return out, pad_array


def unpad(img, pad):
    if pad[2] + pad[3] > 0:
        img = img[:, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        img = img[:, :, :, pad[0]:-pad[1]]
    return img
