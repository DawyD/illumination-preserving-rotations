"""
This code is based on https://github.com/conscienceli/IterNet/ which is released under the follwing licese:

MIT License

Copyright (c) 2019 conscienceli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU, BatchNormalization, Activation
from tensorflow.keras.models import Model

from models.net import Net


class IterNet(Net):
    def __init__(self, input_shape, nr_feats=32, drop_rate=0.1, activation=ReLU, iteration=3):
        super(IterNet).__init__()

        self.input_shape = input_shape
        self.nr_feats = nr_feats
        self.drop_rate = drop_rate
        self.activation = activation
        self.iteration = iteration

    def get_name(self):
        return "IterNet-f{f}d{drop}{iter}".format(
            f=self.nr_feats,
            drop=self.drop_rate,
            iter=self.iteration)

    def net(self):
        inputs = Input(self.input_shape)
        conv1 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats, (3, 3), padding='same')(inputs)))
        conv1 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats, (3, 3), padding='same')(conv1)))
        a = conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 2, (3, 3), padding='same')(pool1)))
        conv2 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 2, (3, 3), padding='same')(conv2)))
        b = conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 4, (3, 3), padding='same')(pool2)))
        conv3 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 4, (3, 3), padding='same')(conv3)))
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 8, (3, 3), padding='same')(pool3)))
        conv4 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 8, (3, 3), padding='same')(conv4)))
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 16, (3, 3), padding='same')(pool4)))
        conv5 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 16, (3, 3), padding='same')(conv5)))

        up6 = concatenate([Conv2DTranspose(self.nr_feats * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4],
                          axis=3)
        conv6 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 8, (3, 3), padding='same')(up6)))
        conv6 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 8, (3, 3), padding='same')(conv6)))

        up7 = concatenate([Conv2DTranspose(self.nr_feats * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3],
                          axis=3)
        conv7 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 4, (3, 3), padding='same')(up7)))
        conv7 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 4, (3, 3), padding='same')(conv7)))

        up8 = concatenate([Conv2DTranspose(self.nr_feats * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                          axis=3)
        conv8 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 2, (3, 3), padding='same')(up8)))
        conv8 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats * 2, (3, 3), padding='same')(conv8)))

        up9 = concatenate([Conv2DTranspose(self.nr_feats, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats, (3, 3), padding='same')(up9)))
        conv9 = Dropout(self.drop_rate)(self.activation()(Conv2D(self.nr_feats, (3, 3), padding='same')(conv9)))

        pt_conv1a = Conv2D(self.nr_feats, (3, 3), padding='same')
        pt_activation1a = self.activation()
        pt_dropout1a = Dropout(self.drop_rate)
        pt_conv1b = Conv2D(self.nr_feats, (3, 3), padding='same')
        pt_activation1b = self.activation()
        pt_dropout1b = Dropout(self.drop_rate)
        pt_pooling1 = MaxPooling2D(pool_size=(2, 2))

        pt_conv2a = Conv2D(self.nr_feats * 2, (3, 3), padding='same')
        pt_activation2a = self.activation()
        pt_dropout2a = Dropout(self.drop_rate)
        pt_conv2b = Conv2D(self.nr_feats * 2, (3, 3), padding='same')
        pt_activation2b = self.activation()
        pt_dropout2b = Dropout(self.drop_rate)
        pt_pooling2 = MaxPooling2D(pool_size=(2, 2))

        pt_conv3a = Conv2D(self.nr_feats * 4, (3, 3), padding='same')
        pt_activation3a = self.activation()
        pt_dropout3a = Dropout(self.drop_rate)
        pt_conv3b = Conv2D(self.nr_feats * 4, (3, 3), padding='same')
        pt_activation3b = self.activation()
        pt_dropout3b = Dropout(self.drop_rate)

        pt_tranconv8 = Conv2DTranspose(self.nr_feats * 2, (2, 2), strides=(2, 2), padding='same')
        pt_conv8a = Conv2D(self.nr_feats * 2, (3, 3), padding='same')
        pt_activation8a = self.activation()
        pt_dropout8a = Dropout(self.drop_rate)
        pt_conv8b = Conv2D(self.nr_feats * 2, (3, 3), padding='same')
        pt_activation8b = self.activation()
        pt_dropout8b = Dropout(self.drop_rate)

        pt_tranconv9 = Conv2DTranspose(self.nr_feats, (2, 2), strides=(2, 2), padding='same')
        pt_conv9a = Conv2D(self.nr_feats, (3, 3), padding='same')
        pt_activation9a = self.activation()
        pt_dropout9a = Dropout(self.drop_rate)
        pt_conv9b = Conv2D(self.nr_feats, (3, 3), padding='same')
        pt_activation9b = self.activation()
        pt_dropout9b = Dropout(self.drop_rate)

        conv9s = [conv9]
        outs = []
        a_layers = [a]
        for iteration_id in range(self.iteration):
            out = Conv2D(1, (1, 1), activation='sigmoid', name=f'out1{iteration_id + 1}')(conv9s[-1])
            outs.append(out)

            conv1 = pt_dropout1a(pt_activation1a(pt_conv1a(conv9s[-1])))
            conv1 = pt_dropout1b(pt_activation1b(pt_conv1b(conv1)))
            a_layers.append(conv1)
            conv1 = concatenate(a_layers, axis=3)
            conv1 = Conv2D(self.nr_feats, (1, 1), padding='same')(conv1)
            pool1 = pt_pooling1(conv1)

            conv2 = pt_dropout2a(pt_activation2a(pt_conv2a(pool1)))
            conv2 = pt_dropout2b(pt_activation2b(pt_conv2b(conv2)))
            pool2 = pt_pooling2(conv2)

            conv3 = pt_dropout3a(pt_activation3a(pt_conv3a(pool2)))
            conv3 = pt_dropout3b(pt_activation3b(pt_conv3b(conv3)))

            up8 = concatenate([pt_tranconv8(conv3), conv2], axis=3)
            conv8 = pt_dropout8a(pt_activation8a(pt_conv8a(up8)))
            conv8 = pt_dropout8b(pt_activation8b(pt_conv8b(conv8)))

            up9 = concatenate([pt_tranconv9(conv8), conv1], axis=3)
            conv9 = pt_dropout9a(pt_activation9a(pt_conv9a(up9)))
            conv9 = pt_dropout9b(pt_activation9b(pt_conv9b(conv9)))

            conv9s.append(conv9)

        out2 = Conv2D(1, (1, 1), activation='sigmoid', name='final_out')(conv9)
        outs.append(out2)

        model = Model(inputs=[inputs], outputs=outs)

        return model
