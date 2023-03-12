"""

"""

from EAMSTCN.decoder import FpnDecoder, StcnDecoder, Decoder1, Decoder2
from EAMSTCN.KeyEncoder import KeyEncoder
from EAMSTCN.ValEncoder import ValEncoder
from EAMSTCN.ResNets import ResKeyEncoder, ResValueEncoder
from EAMSTCN.CustomEncoder import CustomEncoder
from EAMSTCN.modules import FeatureFusionBlock
from EAMSTCN.modules import ProjectToKey
from EAMSTCN.memory import *
from EAMSTCN.utils import *


class EamStcn(nn.Module):
    """"""

    def __init__(self,
                 key_encoder_model: 'b0',
                 value_encoder_model: 'b0',
                 key_enc_channels=3,
                 value_enc_channels=5,
                 width_coefficient=None,
                 depth_coefficient=None,
                 feature_expansion=False,  # Set to True to add an extra block expanding 1/16 feature map
                 expansion_ch=512,  # Number of channels to expand the feature set to.
                 key_pretrained=False,
                 value_pretrained=False,
                 activation='silu',
                 activation_kwargs=None,
                 in_spatial_shape=None,
                 bias=False,
                 drop_connect_rate=0.2,
                 dropout_rate=None,
                 bn_epsilon=1e-3,
                 bn_momentum=0.01,
                 progress=False,
                 train=False,
                 num_objs=1,
                 memory_top_k=20,
                 max_memory_size=None,
                 ck=64,
                 cv=512,
                 stage_channels=(64, 256, 512, 1024),  # Only used for custom  encoder
                 stage_depth=(3, 4, 6, 3),
                 fpn_stage=2,
                 decoder='decoder1',
                 device='cpu'
                 ):
        super().__init__()
        self.query_key = None
        self.query_f16_features = None
        self.value_key_dim = cv  # Cv Number of dimensions of the value key; 512 default as per STM/STCN papers
        self.key_proj_dim = ck  # Ck Number of dimensions of the query key; 64 default as per STCN papers
        self.value_key = None
        self.device = device

        if decoder == 'decoder1' or decoder == 'decoder2' or key_encoder_model == 'resnet':
            self.upres_decoder = True
        elif decoder == 'fpn':
            self.upres_decoder = False
        else:
            print('Decoder choice must be one of: "decoder1", "decoder2" or "fpn"\n"decoder1" selected as default')
            decoder = 'decoder1'
            self.upres.decoder = True

        if value_encoder_model == 'resnet':
            self.value_encoder = ResValueEncoder()
            self.value_out_channels = 256

        else:
            self.value_encoder = ValEncoder(
                value_encoder_model, value_enc_channels, in_spatial_shape, activation, activation_kwargs,
                bias, drop_connect_rate, dropout_rate, bn_epsilon, bn_momentum, value_pretrained, progress)
            self.value_out_channels = self.value_encoder.cfg['out_channel'][-1]

        if key_encoder_model == 'resnet':
            self.key_encoder = ResKeyEncoder()
            self.key_out_channels = 1024  # Take dimension of the final stage
            # if upres_decoder:
            #     self.compress = nn.Conv2d(1024, 512, kernel_size=3, padding='same')

        elif key_encoder_model == 'custom':
            self.key_encoder = CustomEncoder(stage_depth, stage_channels)
            self.key_out_channels = stage_channels[-1]  # Take dimension of the final stage

        elif feature_expansion:
            self.key_encoder = KeyEncoder(
                key_encoder_model, key_enc_channels, in_spatial_shape, activation, activation_kwargs,
                bias, drop_connect_rate, dropout_rate, bn_epsilon, bn_momentum, key_pretrained, progress,
                width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                feature_expansion=True, expansion_ch=expansion_ch)
            # Take dimension of the final stage
            self.key_out_channels = self.key_encoder.cfg['out_channel'][-1] if not feature_expansion else expansion_ch

        else:
            self.key_encoder = KeyEncoder(
                key_encoder_model, key_enc_channels, in_spatial_shape, activation, activation_kwargs,
                bias, drop_connect_rate, dropout_rate, bn_epsilon, bn_momentum, key_pretrained, progress,
                width_coefficient=width_coefficient, depth_coefficient=depth_coefficient)
            # Take dimension of the final stage
            self.key_out_channels = self.key_encoder.cfg['out_channel'][-1] if not feature_expansion else expansion_ch

        if self.upres_decoder:
            self.compress = nn.Conv2d(self.key_out_channels, 512, kernel_size=3, padding='same')

        # Select which memory bank is required
        self.memory = TrainMemoryBank() if train else MemoryBank(num_objs, memory_top_k, max_memory_size)

        # Merges the 1/16 feature maps of the query and value encoders
        self.feature_fusion = FeatureFusionBlock(self.key_out_channels + self.value_out_channels, self.value_key_dim)

        self.key_projection = ProjectToKey(self.key_out_channels,
                                           self.key_proj_dim)  # Out channels (Ck) set to 64 as per STCN paper

        # Get the number of output channels of each feature map for the FPN network. The active feature stages of the
        # encoders can be set with the cfg dict 'is_feature_stage' flags in KeyEncoder.py
        # First update the final number of out channels, since the FPN decoder receives memory value (dim 512)
        # merged with the query key 1/16 features

        if key_encoder_model == 'resnet':
            # Swap the decoder declaration depending on which FPN is used
            if self.upres_decoder:
                self.decoder = StcnDecoder([64, 256, 512, 1024])
            else:
                self.decoder = FpnDecoder([64, 256, 512, 1024 + self.value_key_dim], 256, fpn_stage)

        elif key_encoder_model == 'custom':
            if self.upres_decoder:
                self.decoder = Decoder1([64, 192, 384, 1024]) if decoder == 'decoder1' else \
                    Decoder2([64, 192, 384, 1024])
            else:
                self.decoder = FpnDecoder([*stage_channels[0:-1], stage_channels[-1] + self.value_key_dim], 256,
                                          fpn_stage)
            # self.key_f16_compress = nn.Conv2d(self.key_out_channels, self.key_comp_dim, kernel_size=3, padding='same')

        else:
            if feature_expansion:
                self.key_encoder.cfg['out_channel'][-1] = self.value_key_dim + 512 if self.upres_decoder else \
                    self.value_key_dim + expansion_ch
            else:
                self.key_encoder.cfg['out_channel'][-1] = self.value_key_dim + (
                    512 if self.upres_decoder else self.key_encoder.cfg['out_channel'][-1])

            encoder_out_channels = zip(self.key_encoder.cfg['out_channel'], self.key_encoder.cfg['is_feature_stage'])
            # Set the FPN out channels to the same as the key decoder's dimensions
            if self.upres_decoder:
                self.decoder = \
                    Decoder1([ch for ch, feature in encoder_out_channels if feature]) if decoder == 'decoder1' else \
                    Decoder2([ch for ch, feature in encoder_out_channels if feature])
            else:
                self.decoder = FpnDecoder([ch for ch, feature in encoder_out_channels if feature],
                                          self.key_out_channels,
                                          fpn_stage)

    def get_features(self, x):
        """ Method o obtain the stage features for observation
        :param x: A suitably encoded image tensor
        :return: A dictionary of the encoded features from the end of each of the main CNN blocks
        """
        return self.key_encoder.get_features(x)

    def read_mem(self, query_key):
        return self.memory.match_memory(query_key)

    def get_mask_for_frame(self, query_features, query_key):
        """Concatenate the key's final stage  features with memory"""
        num_objs = self.memory.num_objects
        key_f16 = query_features['stage_4'].expand(num_objs, -1, -1,
                                                   -1)  # expands the key features for the number of objects
        if self.upres_decoder:
            key_f16 = self.compress(key_f16)
        final_value = torch.cat([self.read_mem(query_key), key_f16], 1)
        query_features['stage_4'] = final_value

        return torch.sigmoid(self.decoder(query_features))

    def clear_mem(self):
        self.memory.clear()

    def eval_encode_value(self, frame, mask):
        """Extract memory key/value for a frame with multiple masks
        view same data in form of . Repeat copies data by the index number
        frame  is a copy and  resized to (k, 3, h, w)
        :param frame:
        :param mask:
        :return:
        """
        num_objs, _, H, W = mask.shape
        frame = frame.view(1, 3, H, W).repeat(num_objs, 1, 1, 1)
        remainder = get_other_objects_mask(mask)
        key_f16_features = self.query_f16_features.repeat(num_objs, 1, 1, 1)
        value_features = self.value_encoder(torch.cat([frame, mask, remainder], 1))
        return self.feature_fusion(value_features['stage_4'], key_f16_features).unsqueeze(2)

    def eval_encode_key(self, frame):
        """"""
        query_features = self.key_encoder(frame)
        self.query_f16_features = query_features['stage_4'].to(self.device)
        query_key = self.key_projection(query_features['stage_4']).to(self.device)
        return query_features, query_key

    def save_frame(self, prev_frame, prev_mask, temp=False):
        """

        :param prev_frame:
        :param prev_mask:
        :param temp:
        :return:
        """
        prev_key = self.query_key
        value_features = self.value_encoder(torch.cat([prev_frame, prev_mask], 1))
        value_key = self.feature_fusion(value_features['stage_4'], self.query_f16_features)

        self.memory.add_to_memory(prev_key, value_key, temp)

    @staticmethod
    def reshape_(key_features, query_key, B, T):
        """ Method used when training on sequences of 3 images to reshape encoder results back to B*T*C*H*W
        :param self:
        :param T: int: frames in the image sequence
        :param B: int: batch size
        :param query_key: tensor of query keys to be reshaped
        :param key_features: dict  of query features to be reshaped
        :return: reshaped query_key and key_features
        """
        query_key = query_key.view(B, T, *query_key.shape[-3:]).transpose(1, 2).contiguous()
        key_features['stage_1'] = key_features['stage_1'].view(B, T, *key_features['stage_1'].shape[-3:])
        key_features['stage_2'] = key_features['stage_2'].view(B, T, *key_features['stage_2'].shape[-3:])
        key_features['stage_3'] = key_features['stage_3'].view(B, T, *key_features['stage_3'].shape[-3:])
        key_features['stage_4'] = key_features['stage_4'].view(B, T, *key_features['stage_4'].shape[-3:])
        return key_features, query_key

    def predict_mask(self, key_features, value_memory_Vm, query_key, fr_num, obj_selector, num_objs=2):
        """ Method used when training on sequences of 3 images to predict the obj mask.
        Emulates the procedure used in evaluation mode but on batches on sequences instead of a single one.
        :param obj_selector:
        :param query_key:  tensor : batch of encoded query keys
        :param key_features: tensor: batch of feature sets from the key encoder
        :param value_memory_Vm: first frame encoded value and then the first and second frame for the final prediction
        :param fr_num: int: 1 or 2: The frame we are segmenting (NB the very first frame is 0)
        :param num_objs: int: the number of masked objects
        :return: The raw logits and the softmax probability"""

        # memory should be a TrainMemoryBank
        if not isinstance(self.memory, TrainMemoryBank):
            raise Exception("This method is for training only. Set model train flag to instantiate training model")

        # query key shape is B, Ck, T, H, W Therefore we send B,Ck, 1 or 2 , H, W for the memory
        # and B, Ck, H, W for the frame specific(1 or 2) Query key
        affinity = self.memory.affinity(query_key[:, :, 0:fr_num], query_key[:, :, fr_num])

        # frame cnn stage features for the decoder -
        fr_feats = {'stage_1': key_features['stage_1'][:, fr_num], 'stage_2': key_features['stage_2'][:, fr_num],
                    'stage_3': key_features['stage_3'][:, fr_num], 'stage_4': key_features['stage_4'][:, fr_num]}

        if num_objs == 1:  # Just in case design is changed - branch never used
            fr_feats['stage_4'] = self.memory.match_memory(affinity, value_memory_Vm, fr_feats['stage_4'])
            raw_logits = self.decoder(fr_feats)
            mask_prob = torch.sigmoid(raw_logits)
        else:
            f16_key_features = self.compress(fr_feats['stage_4']) if self.upres_decoder else fr_feats['stage_4']
            fr_feats['stage_4'] = self.memory.match_memory(affinity, value_memory_Vm[:, 0], f16_key_features)
            obj0_raw_logits = self.decoder(fr_feats)
            fr_feats['stage_4'] = self.memory.match_memory(affinity, value_memory_Vm[:, 1], f16_key_features)
            obj1_raw_logits = self.decoder(fr_feats)

            raw_logits = torch.cat([obj0_raw_logits, obj1_raw_logits], dim=1)
            mask_prob = torch.sigmoid(raw_logits)

            # Tensor magic obj_selector shape B 2 1 1
            # The masks for the objects are multiplied by 1 or 0 - we set 2nd mask to all zeros for single objects
            mask_prob = mask_prob * obj_selector.unsqueeze(2).unsqueeze(2)

        logits = aggregate(mask_prob, dim=1)
        # mask = torch.softmax(logits, dim=1)[:, 1:]
        mask = torch.softmax(logits, dim=1)[:, 1:]

        return logits, mask  # prob

    def forward(self, mode, *args, **kwargs, ):
        """If the model is in train mode then frame is a batch of three images. The Query and Value keys
        are encoded and the final value is added to the memory. Pre"""
        if mode == 'encode key features':
            # input tensor shape B*T*C*H*W
            frames, B, T = args[0], *args[0].shape[:2]

            # The encoder expects B*C*H*W so flatten to have a batch equivalent of B*T
            self.query_f16_features = self.key_encoder(frames.flatten(start_dim=0, end_dim=1))
            self.query_key = self.key_projection(self.query_f16_features['stage_4'])
            # Reshape the results to B*T*C*H*W
            return self.reshape_(self.query_f16_features, self.query_key, B, T)

        elif mode == 'encode value key':
            frame, mask, key_f16_features = args
            frame_with_mask = torch.cat([frame, mask], 1)
            self.value_key = self.value_encoder(frame_with_mask)['stage_4']
            fused_value_key = self.feature_fusion(self.value_key, key_f16_features)
            return fused_value_key.unsqueeze(2)

        elif mode == 'predict mask':
            return self.predict_mask(*args)
