import tensorflow as tf
from keras.utils import get_file

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

import platform

from keras.layers import *
from keras.models import *
from keras.backend import *
from keras_cv.models import *

WEIGHTS_PATH_NO_TOP_18 = 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5'

#  filters = [64, 128, 256, 512, 1024]
#  filters = [32, 64, 128, 256, 512]

class UNet():
    def __init__(self, input_shape= (224,224,3), model = '18', weights=None, layer_to_freeze=None, num_class=1, dropout=.0, dil_rate = 1):
        self.input_shape = input_shape
        self.model = model
        self.weights = weights
        self.layer_to_freeze = layer_to_freeze
        self.num_class = num_class
        self.dropout = dropout
        self.dil_rate = dil_rate

        self.filters = []
    
    def __call__(self):
        # _SO = platform.system()
        # if _SO == "Darwin":
        #     self.filters = [16, 32, 64, 128, 256]
        # else:
        #     self.filters = [32, 64, 128, 256, 512]


        self.filters = [8, 16, 32, 64, 128]

        if self.model == 'vgg66':
            model = self.vgg()
        else:
            model = self.unet()

        return model

    def unet(self):
        '''Build the unet model'''
        def conv_block(x, num_filters, name):
            '''Convolution Block'''
            x = Conv2D(num_filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(num_filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu", name=name)(x)

            return x

        input_shape = self.input_shape
        filters     = self.filters
        len_f = len(filters[:-1])
        
        inputs = Input(input_shape)

        skip_x = []
        x = inputs

        # -------------Encoder--------------
        for i, f in enumerate(filters[:-1]):
            x = conv_block(x, f, f'enc_{i+1}')
            skip_x.append(x)
            x = MaxPool2D((2, 2))(x)

        # -------------Bridge--------------
        x = conv_block(x, filters[-1], 'bottleneck')

        filters.reverse()
        skip_x.reverse()

        # -------------Decoder--------------
        for i, f in enumerate(filters[:-1]):
            x = UpSampling2D((2, 2))(x)
            xs = skip_x[i]
            x = Concatenate()([x,xs])
            x = conv_block(x, f, f'dec_{len_f-i}')

        # --------------Output---------------
        output = Conv2D(1, (1, 1), padding="same")(x)
        output = Activation("sigmoid")(output)

        return Model(inputs, output)

    def vgg(self):
        '''Build the unet model'''

        # -------------Encoder--------------
        base_VGG = VGG16(include_top = False,
                        weights     = self.weights,
                        input_shape = self.input_shape)

        conv1 = base_VGG.get_layer("block1_conv2").output
        conv2 = base_VGG.get_layer("block2_conv2").output
        conv3 = base_VGG.get_layer("block3_conv3").output
        conv4 = base_VGG.get_layer("block4_conv3").output

        base_VGG.get_layer("block1_conv2")._name = 'enc_1'
        base_VGG.get_layer("block2_conv2")._name = 'enc_2'
        base_VGG.get_layer("block3_conv3")._name = 'enc_3'
        base_VGG.get_layer("block4_conv3")._name = 'enc_4'

        if self.weights is not None:
            for layer in base_VGG.layers:
                layer.trainable = False
                if layer.name == self.layer_to_freeze:
                    break
            
        # -------------Bridge--------------
        bridge = base_VGG.get_layer("block5_conv3").output

        # -------------Decoder--------------
        up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
        concat_1 = concatenate([up1, conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
        concat_2 = concatenate([up2, conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
        concat_3 = concatenate([up3, conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
        concat_4 = concatenate([up4, conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        # --------------Output---------------
        output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        return Model(base_VGG.input, output)