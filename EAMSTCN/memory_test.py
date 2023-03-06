import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
from memory import MemoryBank


# TODO use sigmoid with preds
if __name__ == '__main__':
    from utils import *
    # start a memory
    test_mem = MemoryBank(5, top_k=20, max_size=None)

    W = 12
    H = 12

    # define fake networks
    query = torch.nn.Conv2d(3, 2, 3, padding='same', stride=1)
    query.eval()
    value = torch.nn.Conv2d(5, 2, 3, padding='same', stride=1)  # 4 or 5 channels depending on the masks required
    value.eval()

    # load some images these have five objects
    im = [
        np.array(Image.open("mem_im/fish_key.jpg").resize((W, H))),
        np.array(Image.open("mem_im/dog_key.jpg").resize((H, W))),
        # np.array(Image.open("mem_im/gorilla_key.jpg").resize((H, W))).astype('uint8'),
        # np.array(Image.open("mem_im/polar_key.jpg").resize((H, W))).astype('uint8'),
        # np.array(Image.open("mem_im/truck_key.jpg").resize((H, W))).astype('uint8')
    ]
    mk = [
        np.array(Image.open("mem_im/fish_mask.png").resize((W, H))).astype('uint8'),
        np.array(Image.open("mem_im/dog_mask.png").resize((H, W))).astype('uint8'),
        # np.array(Image.open("mem_im/gorilla_mask.png").resize((H, W))).astype('uint8'),
        # np.array(Image.open("mem_im/polar_mask.png").resize((H, W))).astype('uint8'),
        # np.array(Image.open("mem_im/bear_mask.png").resize((H, W))).astype('uint8')
    ]
    msks = []
    cr = np.array(Image.open("mem_im/car_race.png").resize((3, 3))).astype('uint8')
    print("car race\n", cr)
    # fake evaluation
    for idx, i in enumerate(im):


        if idx == 1:  # fudge just for these images with the small size HW chosen
            mk[idx][0, :4] = 1
            mk[idx][-4:, -1] = 3

        # get the number of objects
        labels = np.unique(mk[idx])
        labels = labels[labels != 0]
        print('labels: ', labels)
        num_objs = len(labels)

        # Read the frame
        frame = torch.from_numpy(i / 255).permute(2, 0, 1).reshape(1, 3, H, W).repeat(num_objs, 1, 1, 1).float()
        print('frame shape ', frame.shape)
        print('Semantic mask \n', mk[idx])

        # Read the mask
        # msk = torch.from_numpy(one_hot_mask(mk[idx], labels)) # must be float added later for simple  print view
        msk = one_hot_mask(torch.from_numpy((mk[idx])).unsqueeze(0))
        print('one hot', msk.shape)
        msks.append(msk.squeeze())
        # print('one hot mask\n', msk)
        # msk = msk.unsqueeze(1)
        # print('msk un-squeezed: ', msk.shape)

        t = 1  # frames size  - not using at the moment
        prob = torch.zeros((num_objs+1, t, 1, H, W))
        # prob[0] = 1e-7
        print(prob.shape)
        print(prob[:, 0].shape)
        prob[:, 0] = aggregate(msk, keep_bg=True)

        # print('aggregate mask\n', prob[4][0])

        # print(f'Bground merged with binary masks {prob.shape} \n', prob)

        # Index 0 - because background not included.
        encode_mask = prob[1:, 0]
        print('mask encode shape', encode_mask.shape)
        print(encode_mask[4].int())  # prints object 4 mask
        # print(encode_mask)
        # encode_mask = msk.float()

        # Produce a mask that is not the object (inverse) . So dim 0 is ones for all the
        # other objects but not object 1
        # Channel 1 is made from the ones of obj1 ... obj n  n!= 2 etc
        if num_objs != 1:

            others = get_other_objects_mask(encode_mask)
        else:
            others = torch.zeros_like(msk)  # fills with zeros but keeps the size of the passed tensor
        # print(f'The others mask has shape {others.shape}. THe mask looks like: \n', others)

        key = query(frame)
        print('key: ', key.shape)
        masks_cat = torch.cat([encode_mask, others], 1)
        f = torch.cat([frame, encode_mask, others], 1)
        ff = torch.cat([frame, masks_cat], 1)  #

        print('are same: ', torch.equal(f, ff))
        print(f'f shape is: {f.shape}')

        v = value(ff)
        print('v: ', v.shape)

        test_mem.add_to_memory(key, v)



    new_frame = np.array(Image.open('mem_im/dog_key.jpg').resize((W, H))).astype('uint8')
    # pred = test_mem.match_memory(query(frame))
    frame = torch.from_numpy(im[0] / 255).permute(2, 0, 1).reshape(1, 3, H, W).repeat(num_objs, 1, 1, 1).float()
    pred = test_mem.match_memory(query(frame))
    # pred = aggregate(pred, keep_bg=True)
    # pred = aggregate(pred, keep_bg=False)
    print(pred.shape)
    print(len(msks))
    print(msks[0].shape, msks[1].shape)
    print(get_other_objects_mask(torch.cat(msks, dim=0)).int())
    # print(torch.argmax(pred, dim=1))
    # prob = aggregate(pred, keep_bg=True)






