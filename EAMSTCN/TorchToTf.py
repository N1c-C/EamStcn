
from EAMSTCN.EamStcn import EamStcn
from utils import *
import onnx
from onnx_tf.backend import prepare
from torch.autograd import Variable
import tensorflow as tf

DEVICE = 'mps'
# Choose one or the other methods to load model
WEIGHTS_PATH = "/Users/Papa/Downloads/b3_b2_fpn2_ck64_phase2_youtube19_no_amp_start_from_p1_88.pth.tar"

model = EamStcn(
    key_encoder_model='b3',
    value_encoder_model='b2',
    key_pretrained=True,
    value_pretrained=True,
    in_spatial_shape=(480, 854),
    train=False,
    ck=64,
    fpn_stage=2,
    device=DEVICE
)

# model.load_state_dict(torch.load(WEIGHTS_PATH))
load_checkpoint_cuda_to_device(WEIGHTS_PATH, model, DEVICE)

dummy_input = Variable(torch.randn(1, 5, 480, 854))
torch.onnx.export(model.value_encoder, dummy_input, '/Users/Papa/onnx/b3_b2_value.onnx')


onnx_model = onnx.load('/Users/Papa/onnx/b3_b2_value.onnx')
tf_rep = prepare(onnx_model)


# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')


print(tf_rep.tensor_dict)

# pic = np.array(Image.open("/Users/Papa/trainval/JPEGImages/480p/pigs/00074.jpg"))
#
# frame = torch.from_numpy(pic / 255).permute(2, 0, 1).reshape(1, 3, 480, 854).float()

x = tf_rep.run(dummy_input)
print(x[3].shape)

# print(x)

tf_rep.export_graph('/Users/Papa/tf_graph/b3_b2_value')
#
# # tar -czvf potato_model.tar.gz ./potato_tensor

converter = tf.lite.TFLiteConverter.from_frozen_graph(
        "%s/mnist.pb" % sys.argv[2], tf_rep.inputs, tf_rep.outputs)
tflite_model = converter.convert()
open("%s/mnist.tflite" % sys.argv[2], "wb").write(tflite_model)


# from tensorflow import keras
# model = keras.models.load_model('/Users/Papa/tf_graph/b3_b2_key.tar.gz')
