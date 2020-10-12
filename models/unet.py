import math as m
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, SeparableConv2D, Input, SpatialDropout2D, MaxPool2D, Concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.regularizers import l1, l2
from models.net import Net
from layers.kerasGroupNorm import GroupNormalization

class UNet(Net):
    def __init__(self, input_shape, out_channels=1, nr_feats=64, nr_blocks=4, nr_conv=2, upscale="nn", drop_rate=0,
                 last_activation="sigmoid", normalization="group", nr_groups=-1, conv_type="full", factor=2.0,
                 name="Unet", initializer="truncated_normal", weight_decay=0.00001, weight_decay_type="l1",
                 filename=None, basename=None):
        """
        Constructs a U-Net network given the parameters specified. If filename is specified, model is loaded instead.
        :param input_shape: Input shape (e.g. (None, None, 3))
        :param out_channels: Number of output channels (e.g. 1)
        :param nr_feats: Number of feature channels for the first conv. block
        :param nr_blocks: Number of convolutional blocks
        :param nr_conv: Number of convolutions in a block
        :param upscale: Either 'nn' for nearest neighbour upsampling followed by 1x1 conv or 'deconv' for deconvolutional upsampling
        :param drop_rate: Rate with which each an entire channel of each convolutional layer is dropped out
        """
        super(UNet).__init__()
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.nr_feats = nr_feats
        self.nr_blocks = nr_blocks
        self.nr_conv = nr_conv
        self.upscale = upscale
        self.drop_rate = drop_rate
        self.last_activation = last_activation
        self.normalization = normalization
        self.nr_groups = nr_groups
        self.conv_type = conv_type
        self.factor = factor
        self.name = name
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.weight_decay_type = weight_decay_type

        self.use_bias = self.normalization is None

        if self.conv_type == "full":
            self.conv = Conv2D
        elif self.conv_type == "separable":
            self.conv = SeparableConv2D
        else:
            raise ValueError("Invalid convolution type, use 'full' or 'separable'")

        if self.weight_decay_type == "l1":
            self.regularizer = l1(self.weight_decay)
        elif self.weight_decay_type == "l2":
            self.regularizer = l2(self.weight_decay)
        elif self.weight_decay_type is None:
            self.regularizer = None
        else:
            raise ValueError("`Unknown weight decay type")

        self.basename = basename
        if filename is not None:
            self.model = tf.keras.models.load_model(filename, compile=False, custom_objects={'GroupNormalization': GroupNormalization})
        else:
            self.model = self.net()

    def net(self):
        """
        Constructs the keras model based on the parameters specified in __init__()
        :return: keras model
        """
        inp = Input(self.input_shape, name='input_image')
        net = inp

        nr_feats = self.nr_feats

        outputs = []
        levels = []
        for _ in range(self.nr_blocks):
            for _ in range(self.nr_conv):
                net = self.conv_fn(net, nr_feats)
                net = self.normalization_fn(net)
                net = ReLU()(net)
                if self.drop_rate != 0:
                    net = tf.keras.layers.SpatialDropout2D(self.drop_rate)(net)
            levels.append(net)
            net = MaxPool2D(2)(net)
            nr_feats = m.ceil(nr_feats * self.factor)

        for _ in range(self.nr_conv):
            net = self.conv_fn(net, nr_feats)
            net = self.normalization_fn(net)
            net = ReLU()(net)
            if self.drop_rate != 0:
                net = SpatialDropout2D(self.drop_rate)(net)

        for _ in range(self.nr_blocks):
            nr_feats = int(nr_feats // self.factor)
            net = self.upscale_fn(net, nr_feats)
            net = Concatenate(axis=-1)([net, levels.pop()])

            for _ in range(self.nr_conv):
                net = self.conv_fn(net, nr_feats)
                net = self.normalization_fn(net)
                net = ReLU()(net)
                if self.drop_rate != 0:
                    net = SpatialDropout2D(self.drop_rate)(net)
            outputs.append(net)

        last = Conv2D(self.out_channels, 1, padding="same", use_bias=True, activation=self.last_activation,
                      kernel_initializer=self.initializer_fn(self.out_channels),
                      kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(net)

        return tf.keras.Model(inputs=inp, outputs=last, name=self.name)

    def normalization_fn(self, net):
        if self.normalization == "group":
            return GroupNormalization(groups=self.nr_groups, scale=False)(net)
        elif self.normalization == "batch":
            return BatchNormalization(scale=False)(net)
        elif self.normalization is None:
            return net
        else:
            raise ValueError("only batch and group normalizations are supported at the moment")

    def initializer_fn(self, nr_features):
        if self.initializer == "truncated_normal":
            stddev = np.sqrt(2 / (9 * nr_features))
            return tf.keras.initializers.TruncatedNormal(stddev=stddev)
        else:
            return self.initializer

    def upscale_fn(self, net, nr_features):
        net_shape = tf.shape(net)  # .as_list()
        if self.upscale == "nn":
            net = tf.image.resize(net, (2 * net_shape[1], 2 * net_shape[2]))
            net = Conv2D(nr_features, 1, padding="same", use_bias=self.use_bias,
                         kernel_initializer=self.initializer_fn(nr_features),
                         kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(net)
            return ReLU()(net)
        elif self.upscale == "deconv":
            net = Conv2DTranspose(nr_features, 2, strides=2, padding='same', use_bias=self.use_bias,
                                  kernel_initializer=self.initializer_fn(nr_features),
                                  kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(net)
            net = self.normalization_fn(net)
            return ReLU()(net)
        else:
            raise ValueError("invalid upscale parameter, use 'nn' or 'deconv'")

    def conv_fn(self, net, nr_features):
        return self.conv(nr_features, 3, padding="same", use_bias=self.use_bias,
                        kernel_initializer=self.initializer_fn(nr_features),
                        kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(net)

    def get_name(self):
        if self.basename is not None:
            return self.basename

        if self.normalization == "group":
            norm = "GN" + str(self.nr_groups)
        elif self.normalization == "batch":
            norm = "BN"
        else:
            norm = ""

        if self.initializer == "glorot_uniform":
            init = "glorot"
        elif self.initializer == "truncated_normal":
            init = ""
        else:
            init = "??"

        if self.weight_decay is not None and self.weight_decay_type is not None:
            wd = "WD{:}-{:0.0E}".format(self.weight_decay_type, self.weight_decay)
        else:
            wd = ""

        return "UNet-b{b}f{f}c{c}d{d}{ctype}{norm}{up}{fact}{init}{wd}".format(
            b=self.nr_blocks,
            f=self.nr_feats,
            c=self.nr_conv,
            up=self.upscale,
            d=self.drop_rate,
            ctype="sep" if self.conv_type == "separable" else self.conv_type,
            norm=norm,
            fact="F" + str(self.factor) if self.factor != 2 else "",
            init=init,
            wd=wd)

