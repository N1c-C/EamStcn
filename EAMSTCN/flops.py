"""

"""

from multiprocessing import freeze_support
from fvcore.nn import FlopCountAnalysis
import PIL.Image as Image
from torch.utils.data import DataLoader
import time
from EAMSTCN.EamStcn import EamStm
import torch
from tqdm import tqdm  # progress bar
from datasets.DavisEvalDataset import DAVISEvalDataset
from datasets.YTubeTestDataset import YouTubeTestDataset
from evaluate.evaluator_adaptive_save import EvalEamStm
from metrics.measurements import *
from utils import *

if __name__ == '__main__':
    freeze_support()

    DEVICE = 'mps' if torch.backends.mps else 'cpu'


    # MODEL_PATH = '/Users/Papa/Trained Models/EamStm_A.pth'
    # MODEL_PATH = '/Users/Papa/Trained Models/EamStm_B.pth'
    # MODEL_PATH = '/Users/Papa/Trained Models/EamStm_C.pth'
    # MODEL_PATH = '/Users/Papa/Trained Models/EamStm_Comparison.pth'
    MODEL_PATH = '/Users/Papa/Trained Models/EamStm_Control.pth'

    # Choose one or the other methods to load model
    # WEIGHTS_PATH = "/Users/Papa/Trained Models/b1_b1_ex512/b1_b1_stcn_512_256_ex512_phase2_yt19amp_bse_60.pth.tar"
    # WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1_ex512/b1_b1_ex512_customfpn_dec_ck64_phase2_resFuse_amp_82.pth.tar' # 83.6
    # WEIGHTS_PATH = '/Users/Papa/Trained Models/res50_res18/res50_res18_phase2_stcn_dec_&_fuse_ck64_yt19_sq_amp_final.pth.tar' # Control
    WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1_ex512/b1_b1_ex512_MixedFuse_stcn512_256_dec_phase2_yt19_amp_final.pth.tar'  # 82.76
    # WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1/b1_b1_stcn_ck64_phase2_yt19_amp_final.pth.tar'  # 79.16
    # WEIGHTS_PATH = '/Users/Papa/Trained Models/b1_b1_ex512/b1_b1_stcn_dec_ck64_ex512_phase2_yt19_no_amp_64.pth.tar' # 80.48

    # model = EamStm(
    #     key_encoder_model='b1',  # EfficientNetV2 letter, 'resnet' or 'custom'
    #     value_encoder_model='b1',  # EfficientNetV2 letter, 'resnet'
    #     key_pretrained=False,
    #     value_pretrained=False,
    #     in_spatial_shape=(480, 854),
    #     train=False,
    #     ck=64,
    #     activation='silu',
    #     fpn_stage=2,
    #     decoder='decoder1',
    #     width_coefficient=None,  # Scales the width of an EfficientNetV2 key encoder
    #     depth_coefficient=None,  # Scales the depth of an EfficientNetV2 key encoder
    #     feature_expansion=True,  # Adds an extra block to the chosen EfficientNetV2 key encoder
    #     expansion_ch=512,  # The number of output channels for the extra block
    #     stage_channels=(64, 192, 384, 512),  # Custom output Ch for each stage. Only used for custom efficient encoder
    #     stage_depth=(2, 3, 4, 2),  # The number of repeated blocks in each stage default = 3, 4, 6, 3, 64, 256, 512,1024
    #     device=DEVICE
    # )

    # load_checkpoint_cuda_to_device(WEIGHTS_PATH, model, DEVICE)
    print("Alert NOTHING LOADED")
    model = torch.load(MODEL_PATH)


    model.eval().to(DEVICE)
    print(f'Number of parameters: {count_parameters(model)}')
    print(f'Number of trainable parameters: {trainable_parameters(model)}')


imx = torch.rand(1, 3, 480, 864).to(DEVICE)
imy = torch.rand(1, 5, 480, 864).to(DEVICE)
y = model.value_encoder
x = model.key_encoder


features, _ = model.eval_encode_key(imx[0:])
print (features['stage_4'].shape)

flopx = FlopCountAnalysis(x, imx)
flopy = FlopCountAnalysis(y, imy)

value_features = y(imy)
fuse = model.feature_fusion

fuseflop = FlopCountAnalysis(fuse, (value_features['stage_4'], features['stage_4']))


val = model.eval_encode_value(imx[0:], torch.rand(1, 1, 480, 864).to(DEVICE))
print(val.squeeze().shape)

# features['stage_4'] = torch.cat([features['stage_4'], val.squeeze().unsqueeze(0)], dim=1)
print(features['stage_4'].shape)
# decoder flops
z1 = model.decoder.compress
z2 = model.decoder.up_16_8
z3 = model.decoder.up_8_4
z4 = model.decoder.pred



# self.block1 = ResBlock(in_channel, out_channel)
# # self.attention = CBAM(out_channel)
# self.block2 = ResBlock(out_channel, out_channel)


# d = Z1(features['stage_4'])
# d1 = model.decoder.up_16_8(features['stage_3'], d)
# d2 = model.decoder.up_8_4(features['stage_2'], d1)
# d3 = model.decoder.pred(F.relu(d2))

flopdec = FlopCountAnalysis(z4, z3(features['stage_2'], z2(features['stage_3'], z1(features['stage_4']))))

print(f'Flop Count: {(flopx.total() + flopy.total() + fuseflop.total() + flopdec.total()) /1e9:.3f} GFlops')
