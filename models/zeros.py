import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, SeparableConv2D, Input, SpatialDropout2D, MaxPool2D, Concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.regularizers import l1, l2
from models.net import Net
from layers.kerasGroupNorm import GroupNormalization

class Zeros(Net):
    def __init__(self, output_shape):
        super(Zeros).__init__()
        self.output_shape = output_shape

    def net(self):
        model = tf.keras.Sequential()
        model.add(Input(self.output_shape, name='output_shape'))
        model.add(tf.keras.layers.Lambda(lambda x: tf.zeros_like(x)[..., 0:1]))
        return model

    def get_name(self):
        return "Zeros"
