"""

"""

from multiprocessing import freeze_support
from collections import defaultdict
import torch.nn as nn
from PIL import Image
from tqdm import tqdm  # progress bar
from EAMSTCN.metrics.measurements import eval_tensor_iou
from EAMSTCN.utils import *
import torchvision.transforms as T


class Trainer:
    """
    Just a training wrapper which can be developed further if need be. Methods are static functions
    """

    def __init__(self):
        self.lc = LossComputer()
        self.so_far = 0
        pass

    # @staticmethod
    def train(self, train_loader, model, optimiser, loss_fn, scaler, epochs, so_far=0, epochs_left=None, step_lr=None,
              early_stop=True, val_loader=None, patience=5, logger=None, log_img=20, save_img=50, model_save=10,
              device='cpu', name='model', SAVE_IMG_DIR=None, SAVE_MODEL_DIR=None):
        """
        :param step_lr:
        :param name:
        :param epochs_left:
        :param so_far:
        :param model_save:
        :param save_img:
        :param log_img:
        :param SAVE_MODEL_DIR:
        :param SAVE_IMG_DIR:
        :param train_loader:
        :param model:
        :param optimiser:
        :param loss_fn:
        :param scaler:
        :param epochs:
        :param early_stop:
        :param val_loader:
        :param patience:
        :param logger:
        :param device:
        :return:
        """
        self.so_far = so_far
        tls = 0  # initial training loss for display on progress bar
        vls = 0 if val_loader is not None else None  # initial val loss for display on progress bar
        avg_loss = 0
        val_loss_hist = []  # keeps track of previous val_loss values so that early stopping can be evoked
        stop = False  # Early stopping flag
        trn_loss_hist = []  # keep track of train loss so that it can be plotted
        iou = 0

        if epochs_left is not None:
            epochs = epochs_left
        end = self.so_far + epochs_left if epochs_left is not None else self.so_far + epochs
        for epoch in range(self.so_far, end):
            avg_loss = 0
            al = 0
            running_iou = 0
            epoch_mean_iou = 0
            batch_mean_iou = 0
            if stop:
                status_bar.write('Early stopping activated')
                break
            status_bar = tqdm(train_loader, unit="Batch", position=0, leave=True)

            # Train Phase
            status_bar.bar_format = "{l_bar}{bar}| Batch: {n}/{total_fmt} Time: {elapsed} < {remaining}, {rate_fmt}{" \
                                    "postfix} "
            status_bar.set_description(f"Epoch {(epoch + 1):04}")
            status_bar.set_postfix({"Batch loss": tls, "Epoch loss": avg_loss, "Batch J": batch_mean_iou,
                                    "Mean J": epoch_mean_iou})

            for batch_idx, data in enumerate(status_bar):
                gt = data['gt1_seq']
                im = data['seq']
                for key, val in data.items():
                    if type(val) != int and type(val) != dict:
                        data[key] = val.to(device)
                frames = data['seq']
                masks_obj1 = data['gt1_seq']
                masks_obj2 = data['gt2_seq']
                obj_select = data['selector']
                # num_objs = data['num_objs'][0].to(device)

                optimiser.zero_grad()  # zero the gradients for each batch

                # Forward Pass
                # Logits shape B 3 H W,  Mask shape B 2 H W
                logits = {}
                masks = {}
                if device == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits['logits_fr1'], masks['frame1_mask'], logits['logits_fr2'], masks['frame2_mask'] = \
                            Trainer.forward(model, frames, masks_obj1, masks_obj2, obj_select, device)
                else:
                    logits['logits_fr1'], masks['frame1_mask'], logits['logits_fr2'], masks['frame2_mask'] = \
                        Trainer.forward(model, frames, masks_obj1, masks_obj2, obj_select, device)

                # to access individual object masks  index [:, 0:1]  and [:, 1:2]

                ce_losses = self.lc.compute({**data, **logits, **masks}, epoch, device)
                # fr1_uncertainty_loss = torch.linalg.norm(
                #                         self.calc_uncertainty(masks['frame1_mask'])) / masks['frame1_mask'].shape[0]
                # fr2_uncertainty_loss = torch.linalg.norm(
                #                         self.calc_uncertainty(masks['frame2_mask'])) / masks['frame2_mask'].shape[0]
                combined_loss = ce_losses['total_loss']  # + 0.001 * (fr1_uncertainty_loss + fr2_uncertainty_loss)

                # Update Progress Bar
                if batch_idx != 0:
                    avg_loss += combined_loss.item()
                    al = avg_loss / batch_idx
                    batch_mean_iou = (ce_losses['iou'].item() + ce_losses['iou2'].item()) / ce_losses['num_frs']
                    running_iou += batch_mean_iou
                    epoch_mean_iou = running_iou / batch_idx

                status_bar.set_postfix(
                    {"Batch loss": combined_loss.item(), "Epoch loss": al, "Batch J": batch_mean_iou,
                     "Mean J": epoch_mean_iou})

                # backward pass
                if scaler is not None:
                    scaler.scale(combined_loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    combined_loss.backward()
                    optimiser.step()

            if step_lr is not None:
                step_lr.step()

                # if batch_idx in [2,22,42,62,82]:
                #     stop = True
                #     break

            # Log to tensorboard
            if logger is not None:
                # Log training loss
                logger.add_scalar("Training Loss", al, epoch)
                logger.add_scalar("J (IoU)", epoch_mean_iou, epoch)

            if logger is not None:
                # Save an image to tensor board every log_img epochs - 1st frame only
                if not (epoch % log_img) and epoch != 0:
                    logger.add_masks("Predictions", masks['frame1_mask'][:, 0:1], epoch)
                    logger.add_masks("Ground truth", data['gt1_seq'][:, 1], epoch)
                    logger.add_images("Frame", data['seq'][:, 1], epoch)

            if logger is not None:
                logger.flush()
            # if val_loader is not None:
            #     Trainer.validate()
            # vls = 0

            # Save an image to disc every save_img epochs both frames
            if SAVE_IMG_DIR is not None:
                if epoch % save_img == 0 and epoch != 0:
                    # Prepare mask predictions for torch save utility shape B, C, H, W  Where B = B*T
                    pred_image = torch.stack([
                        masks['frame1_mask'].unsqueeze(2)[:, 0],
                        masks['frame2_mask'].unsqueeze(2)[:, 0]],
                        1).flatten(start_dim=0, end_dim=1)
                    os.makedirs(SAVE_IMG_DIR, exist_ok=True)
                    if device == 'mps':
                        pred_image = pred_image.cpu()
                    inv_normalise = T.Compose([
                        T.Normalize(
                            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                        )])

                    try:
                        torchvision.utils.save_image(pred_image[:, ], f'{SAVE_IMG_DIR}pred_{epoch}_{batch_idx}.png')
                        torchvision.utils.save_image(inv_normalise(im[:, 1:].flatten(start_dim=0, end_dim=1)),
                                                     f'{SAVE_IMG_DIR}image_{epoch}_{batch_idx}.png')
                        torchvision.utils.save_image(F.normalize(gt[:, 1:].flatten(start_dim=0, end_dim=1)),
                                                     f'{SAVE_IMG_DIR}gt_{epoch}_{batch_idx}.png')

                    except RuntimeError:
                        print('Image save failed')

                    del pred_image

            # Clear memory
            del logits, masks
            del data
            del masks_obj1
            del masks_obj2
            del obj_select
            del frames

            # test for early stopping: in this case quit if loss has not reduced in the past patience number of tests
            # if len(val_loss_hist) > patience and early_stop:
            #     past_loss = val_loss_hist[-patience:]
            #     stop = all(j >= i for i, j in zip(past_loss, past_loss[1:]))

            # Save state dicts every model_save epochs
            if SAVE_MODEL_DIR is not None:
                if (epoch % model_save) == 0 and epoch != 0:
                    if step_lr is not None:
                        checkpoint = {
                            'state_dict': model.state_dict(),
                            'optimiser': optimiser.state_dict(),
                            'scheduler': step_lr.state_dict()
                        }
                    else:
                        checkpoint = {
                            'state_dict': model.state_dict(),
                            'optimiser': optimiser.state_dict(),
                        }
                    save_checkpoint(checkpoint, SAVE_MODEL_DIR, f'{name}_{epoch}')
            logger.close()

    @staticmethod
    def forward(model, frames, masks_obj1, masks_obj2, obj_select, device):
        """
        :param model:
        :param frames:
        :param masks_1:
        :param masks_2:
        :param obj_select:
        :param device:
        :return:
        """

        # Create an additional other objects mask to add discrimination in the value encoding
        # masks shape is B T C H W   C represents number of objects

        model.train()

        # encode all the keys in one go
        key_features, query_keys = model('encode key features', frames)

        # Use gt image and mask to encode frame 0 memory value
        obj1_fr0_mask = torch.cat([masks_obj1[:, 0], masks_obj2[:, 0]], 1)
        obj2_fr0_mask = torch.cat([masks_obj2[:, 0], masks_obj1[:, 0]], 1)

        obj1_fr0_val = model('encode value key', frames[:, 0], obj1_fr0_mask, key_features['stage_4'][:, 0]).to(device)
        obj2_fr0_val = model('encode value key', frames[:, 0], obj2_fr0_mask, key_features['stage_4'][:, 0]).to(device)
        fr0_val = torch.stack([obj1_fr0_val, obj2_fr0_val], 1)

        # predict frame1 mask by referencing fr0_val
        fr1_logits, fr1_mask = model('predict mask', key_features, fr0_val, query_keys, 1, obj_select)

        # encode value of predicted frame1 and add to memory
        obj1_fr1_mask = torch.cat([fr1_mask[:, 0:1], fr1_mask[:, 1:2]], 1)
        obj2__fr1_mask = torch.cat([fr1_mask[:, 1:2], fr1_mask[:, 0:1]], 1)

        obj1_fr1_val = model('encode value key', frames[:, 1], obj1_fr1_mask, key_features['stage_4'][:, 1]).to(device)
        obj2_fr1_val = model('encode value key', frames[:, 1], obj2__fr1_mask, key_features['stage_4'][:, 1]).to(device)
        fr1_val = torch.stack([obj1_fr1_val, obj2_fr1_val], 1)  # shape B Obj Cv T H W
        val_mem = torch.cat([fr0_val, fr1_val], 3)  # Join along Cv dimension as in inference

        del fr0_val,  # Clear some RAM

        # Predict frame 2 mask by referencing frame0 and frame1 memory values
        fr2_logits, fr2_mask = model('predict mask', key_features, val_mem, query_keys, 2, obj_select)

        # to access individual object masks  index [:, 0:1]  and [:, 1:2]
        return fr1_logits, fr1_mask, fr2_logits, fr2_mask

    @staticmethod
    def validate():
        pass

    @staticmethod
    def calc_uncertainty(masks):
        """https://github.com/AFB-URR/blob/main/myutils/data.py
        Use top to find the two highest probabilities for a pixel from all the objects present and returns a uncertainty score.
        if all probs are 1 and zero the score is 0"""
        # masks shape: b, objs, h, w
        top_prob, _ = masks.topk(k=2, dim=1)
        # We get a two tensor array: [0] has the top probability for a pixel from any of the object channels
        # [1] has the second-highest probability
        uncertainty = top_prob[:, 0] / top_prob[:, 1] + 1e-8  # bs, h, w
        uncertainty = torch.exp(1 - uncertainty)  # bs, 1, h, w
        return uncertainty


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
#
# class BootstrappedCE(nn.Module):
#     def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
#         super().__init__()
#
#         self.start_warm = start_warm
#         self.end_warm = end_warm
#         self.top_p = top_p
#
#     def forward(self, input, target, it=0):
#         if it < self.start_warm:
#             return F.cross_entropy(input, target), 1.0
#
#         raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
#         num_pixels = raw_loss.numel()  # returns number of elements
#
#         if it > self.end_warm:
#             this_p = self.top_p
#         else:
#             this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))
#         loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
#         return loss.mean(), this_p


class BootstrappedCE(nn.Module):

    def __init__(self, start_warm=12, end_warm=45, top_p=0.15):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.start_warm = start_warm  # epochs
        self.end_warm = end_warm  # epochs  - 1800 iterations per epoch
        self.top_p = top_p

    def forward(self, preds, gt, epoch, device):

        if epoch < self.start_warm or device == 'mps':
            return self.ce(preds, gt)

        raw_loss = F.cross_entropy(preds, gt, reduction='none').view(-1)
        num_pixels = raw_loss.numel()  # returns number of elements

        if epoch > self.end_warm:
            this_p = self.top_p
        else:

            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - epoch) / (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean()


class LossComputer:
    """ Assumes the gt is provided as Hard label encoding ie the gt is 1 for obj1, 2 for obj2 e.t.c.
        Shape requirements for hard labels with CELoss is pred: 1 2 H W   cls Gt: 1, H,W
        As a consequence we must loop through the mask predictions and sum the error to determine the batch mean loss
    """

    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        # self.ce = BootstrappedCE()
        self.select = torch.FloatTensor([1, 1])

    def compute(self, data, epoch, device):
        """
        :param data:
        :return:
        """
        sel_test = self.select.to(device)
        losses = defaultdict(int)
        batch, t, _, _, _ = data['gt1_seq'].shape
        selector = data.get('selector', None)
        for fr in range(1, t):
            for idx in range(batch):
                if torch.equal(selector[idx], sel_test):
                    loss = self.ce(data[f'logits_fr{fr}'][idx:idx + 1],
                                   data['cls_gt'][idx:idx + 1, fr])
                else:
                    loss = self.ce(data[f'logits_fr{fr}'][idx:idx + 1, :2],
                                   data['cls_gt'][idx:idx + 1, fr])  # epoch, device
                losses[f'loss_fr{fr}'] += loss / batch
            losses['total_loss'] += losses[f'loss_fr{fr}']

            # find J (iou) as a batch: to access individual object masks  index [:, 0:1]  and [:, 1:2]
            losses['iou'] += eval_tensor_iou(data[f'frame{fr}_mask'][:, 0:1] > 0.5, data['gt1_seq'][:, fr] > 0.5)[0]
            losses['num_frs'] += 1
            if selector is not None:
                losses['iou2'] += eval_tensor_iou(data[f'frame{fr}_mask'][:, 1:2] > 0.5, data['gt1_seq'][:, fr] > 0.5)[
                    0]

        return losses


if __name__ == '__main__':
    freeze_support()
