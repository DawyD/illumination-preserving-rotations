import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, SeparableConv2D, Input, SpatialDropout2D, MaxPool2D, Concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.regularizers import l1, l2
from models.net import Net
from layers.kerasGroupNorm import GroupNormalization


class DnCNN(Net):
    def __init__(self, input_shape, out_channels=1, nr_feats=64, nr_conv=16, last_activation="sigmoid", normalization="group",
              nr_groups=-1, conv_type="full", name="DnCNN", weight_decay=0, weight_decay_type=None, filename=None, basename=None):
        super(DnCNN).__init__()
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.nr_feats = nr_feats
        self.nr_conv = nr_conv
        self.last_activation = last_activation
        self.normalization = normalization
        self.nr_groups = nr_groups
        self.conv_type = conv_type
        self.name = name
        self.weight_decay = weight_decay
        self.weight_decay_type = weight_decay_type

        self.initializer = 'glorot_uniform'

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
        use_bias = True if self.normalization is None else False

        inp = Input(self.input_shape, name='input')
        net = inp
        for layers in range(self.nr_conv):
            net = self.conv(self.nr_feats, 3, padding="same", use_bias=use_bias, kernel_initializer=self.initializer,
                            kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(net)

            if self.normalization == "group":
                net = GroupNormalization(groups=self.nr_groups, scale=False)(net)
            elif self.normalization == "batch":
                net = BatchNormalization(scale=False)(net)
            elif self.normalization is not None:
                raise ValueError("only batch and group normalizations are supported at the moment")
            net = ReLU()(net)

        net = self.conv(self.out_channels, 3, padding='same', use_bias=True, kernel_initializer=self.initializer,
                        kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, name='conv_last',
                        activation=self.last_activation)(net)

        return tf.keras.Model(inputs=inp, outputs=net, name=self.name)

    def get_name(self):
        if self.basename is not None:
            return self.basename

        if self.normalization == "group":
            norm = "GN" + str(self.nr_groups)
        elif self.normalization == "batch":
            norm = "BN"
        else:
            norm = "??"

        if self.weight_decay is not None and self.weight_decay_type is not None:
            wd = "WD" + self.weight_decay_type + "%E".format(self.weight_decay)
        else:
            wd = ""

        return "DnCNN-c{c}f{f}{ctype}{norm}{wd}".format(
            c=self.nr_conv,
            f=self.nr_feats,
            ctype="sep" if self.conv_type == "separable" else self.conv_type,
            norm=norm,
            wd=wd)