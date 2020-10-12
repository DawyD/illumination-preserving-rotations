import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MaxPool2D, Conv2D, Conv2DTranspose, Reshape, Lambda, Input
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.models import Model

from models.net import Net


class FCN(Net):
    def __init__(self, input_shape, weight_decay=32, drop_rate=0.1):
        super(FCN).__init__()

        self.input_shape = input_shape
        self.weight_decay = weight_decay
        self.drop_rate = drop_rate

    def get_name(self):
        return "FCN-wd{wd}d{drop}".format(
            wd=self.weight_decay,
            drop=self.drop_rate)

    def net(self):
        base_model = self.vgg16(self.input_shape, l2=self.weight_decay, dropout=self.drop_rate)
        vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        weight_list = vgg16.get_weights()
        if self.input_shape[-1] % 3 != 0:
            raise ValueError("Input shape has to be divisible by 3 (due to the 1st layer of VGG16)")
        weight_list[0] = np.tile(weight_list[0][:, :, :, :], (1, 1, self.input_shape[-1] // 3, 1)) 
        weight_list[0] /= (self.input_shape[-1] // 3)
        weight_list[26] = weight_list[26].reshape(7, 7, 512, 4096)
        weight_list[28] = weight_list[28].reshape(1, 1, 4096, 4096)
        weight_list[30] = weight_list[30].reshape(1, 1, 4096, 1000)
        base_model.set_weights(weight_list)

        fcn32 = self.fcn32(base_model, l2=self.weight_decay, out_channels=1, activation="sigmoid")
        fcn16 = self.fcn16(base_model, fcn32, l2=self.weight_decay, out_channels=1, activation="sigmoid")
        fcn8 = self.fcn8(base_model, fcn16, l2=self.weight_decay, out_channels=1, activation="sigmoid")

        return fcn8

    @staticmethod
    def vgg16(input_shape=(None, None, 3), l2=0.0, dropout=0.0):
        """
        Convolutionized VGG16 network.

        :param input_shape: Input shape (excluding the batch dimension)
        :param l2: L2 regularization
        :param dropout: Dropout rate
        :return: Keras VGG16 model
        """

        # Input
        input_layer = Input(shape=input_shape, name='input')

        # Preprocessing
        x = Reshape((512, 512, input_shape[-1] // 3, 3))(input_layer)
        x = Lambda(tf.keras.applications.vgg16.preprocess_input, name='preprocessing')(x)
        x = Reshape((512, 512, input_shape[-1]))(x)
        # x = input_layer

        # Block 1
        x = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block1_conv1')(x)
        x = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block1_conv2')(x)
        x = MaxPool2D(name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block2_conv1')(x)
        x = Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block2_conv2')(x)
        x = MaxPool2D(name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block3_conv1')(x)
        x = Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block3_conv2')(x)
        x = Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block3_conv3')(x)
        x = MaxPool2D(name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block4_conv1')(x)
        x = Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block4_conv2')(x)
        x = Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block4_conv3')(x)
        x = MaxPool2D(name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block5_conv1')(x)
        x = Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block5_conv2')(x)
        x = Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='block5_conv3')(x)
        x = MaxPool2D(name='block5_pool')(x)

        # Convolutionized fully-connected layers
        x = Conv2D(4096, 7, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='conv6')(x)
        x = tf.keras.layers.Dropout(rate=dropout, name='drop6')(x)
        x = Conv2D(4096, 1, padding='same', activation='relu', kernel_regularizer=L2(l2=l2), name='conv7')(x)
        x = tf.keras.layers.Dropout(rate=dropout, name='drop7')(x)

        # Inference layer
        x = Conv2D(1000, 1, padding='same', activation='softmax', name='pred')(x)

        return Model(input_layer, x)

    @staticmethod
    def fcn32(vgg16, l2=0.0, out_channels=21, activation='softmax'):
        """
        32x upsampled Fully Convolutional Network.

        :param vgg16: VGG16 base model (encoder)
        :param l2: L2 regularization
        :param out_channels: Number of output channels
        :param activation: Name of the activation function for the last layer
        :return: Keras FCN32 model
        """

        x = Conv2D(out_channels, 1, padding='same', activation='linear', kernel_initializer=Zeros(),
                   kernel_regularizer=L2(l2=l2), name='score7')(vgg16.get_layer('drop7').output)

        x = Conv2DTranspose(out_channels, 64, strides=(32, 32), padding='same', use_bias=False, activation=activation,
                            kernel_initializer=BilinearInitializer(), kernel_regularizer=L2(l2=l2), name='fcn32')(x)

        return Model(vgg16.input, x)

    @staticmethod
    def fcn16(vgg16, fcn32, l2=0.0, out_channels=21, activation='softmax'):
        """
        16x upsampled Fully Convolutional Network.

        :param vgg16: VGG16 base model (encoder)
        :param fcn32: FCN32 model
        :param l2: L2 regularization
        :param out_channels: Number of output channels
        :param activation: Name of the activation function for the last layer
        :return: Keras FCN16 model
        """

        x = Conv2DTranspose(out_channels, 4, strides=(2, 2), padding='same', use_bias=False, activation='linear',
                            kernel_initializer=BilinearInitializer(), kernel_regularizer=L2(l2=l2),
                            name='score7_upsample')(fcn32.get_layer('score7').output)

        y = Conv2D(out_channels, 1, padding='same', activation='linear',
                   kernel_initializer=Zeros(), kernel_regularizer=L2(l2=l2),
                   name='score4')(vgg16.get_layer('block4_pool').output)

        x = tf.keras.layers.Add(name='skip4')([x, y])

        x = Conv2DTranspose(out_channels, 32, strides=(16, 16), padding='same', use_bias=False, activation=activation,
                            kernel_initializer=BilinearInitializer(), kernel_regularizer=L2(l2=l2), name='fcn16')(x)

        return Model(fcn32.input, x)

    @staticmethod
    def fcn8(vgg16, fcn16, l2=0.0, out_channels=21, activation='softmax'):
        """
        8x upsampled Fully Convolutional Network.

        :param vgg16: VGG16 base model (encoder)
        :param fcn16: FCN16 model
        :param l2: L2 regularization
        :param out_channels: Number of output channels
        :param activation: Name of the activation function for the last layer
        :return: Keras FCN8 model
        """

        x = Conv2DTranspose(out_channels, 4, strides=(2, 2), padding='same', use_bias=False, activation='linear',
                            kernel_initializer=BilinearInitializer(), kernel_regularizer=L2(l2=l2),
                            name='skip4_upsample')(fcn16.get_layer('skip4').output)

        y = Conv2D(out_channels, 1, padding='same', activation='linear',
                   kernel_initializer=Zeros(), kernel_regularizer=L2(l2=l2),
                   name='score3')(vgg16.get_layer('block3_pool').output)

        x = tf.keras.layers.Add(name='skip3')([x, y])

        x = Conv2DTranspose(out_channels, 16, strides=(8, 8), padding='same', use_bias=False, activation=activation,
                            kernel_initializer=BilinearInitializer(), kernel_regularizer=L2(l2=l2), name='fcn8')(x)

        return Model(fcn16.input, x)
    

class BilinearInitializer(tf.keras.initializers.Initializer):
    """Initializer for Conv2DTranspose to perform bilinear interpolation on each channel."""

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):

        kernel_size, _, filters, _ = shape

        arr = np.zeros((kernel_size, kernel_size, filters, filters))

        upscale_factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = upscale_factor - 1
        else:
            center = upscale_factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        kernel = (1 - np.abs(og[0] - center) / upscale_factor) * (1 - np.abs(og[1] - center) / upscale_factor)
        # kernel shape is (kernel_size, kernel_size)

        for i in range(filters):
            arr[..., i, i] = kernel

        return tf.convert_to_tensor(arr, dtype=dtype)
