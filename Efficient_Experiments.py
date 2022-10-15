import torch
import torch.nn as nn
from pytorch_efficientnet.efficientnet_v2 import EfficientNetV2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import seaborn as sns

model = EfficientNetV2('b1',
                       in_channels=3,
                       n_classes=50,
                       in_spatial_shape=(224, 224),
                       pretrained=True)

print(model.blocks)
# x - tensor of shape [batch_size, in_channels, image_height, image_width]
x = torch.randn([10, 3, 224, 224])
# print(dir(model))
# to get predictions:
x = Image.open('test/JPEGImages/480p/bear/00000.jpg')
# x = x.resize((224, 224))

plt.imshow(x)
plt.show()
convert = transforms.ToTensor()
x = convert(x)
x = x.reshape(1, 3, 480, 854)
print(f'Bear image shape is {x.shape}')
pred = model(x)
print('out shape:', pred.shape)
# >>> out shape: torch.Size([10, 50])

# to extract features:
features = model.get_features(x)
for i, feature in enumerate(features):
    print('feature %d shape:' % i, feature.shape)
# >>> feature 0 shape: torch.Size([10, 48, 56, 56])
# >>> feature 1 shape: torch.Size([10, 64, 28, 28])
# >>> feature 2 shape: torch.Size([10, 160, 14, 14])
# >>> feature 3 shape: torch.Size([10, 256, 7, 7])
f = model.get_features(x)
f0 = f[5].detach().numpy()
print(f'detached shape is {f0.shape}')


# print(model.blocks[5])


def display_batch(feats, fig_size=(12, 8)):
    print(f'Passed shape {feats[0].shape}')
    """Display a gallery of images"""
    # initialize a figure
    fig = plt.figure(f"level features", figsize=fig_size)
    # loop over the features
    for i, f in enumerate(feats[0]):
        # create a subplot
        if i > 31:
            i = 31
        ax = plt.subplot(4, 8, i + 1)
        # convert img  from channels first to channels last ordering
        # scale pixel intensities to the range [0, 255]
        image = f
        # image = image.transpose((1, 2, 0))
        # image = ((image) * 1).astype("uint8")
        # grab the label id and get the label from the classes list
        # show the image along with the label
        plt.imshow(image, cmap='gray')
        plt.title("Some Features")
        plt.axis("off")
        # sns.heatmap()
    # show the plot
    plt.tight_layout()
    plt.show()


display_batch(f0)
# print(model)

# from efficientnet_pytorch import EfficientNet as EffNet
#
# model2 = EffNet.from_pretrained('efficientnet-b0')

# print(model2)


import tensorflow as tf


# model3 = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation='softmax',
#     include_preprocessing=True
# )
# print(model3.summary())


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


print(MyModule())
print(model)
