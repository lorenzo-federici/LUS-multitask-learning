import tensorflow as tf
from keras.utils import get_file

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

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

        self.filters = [16, 32, 64, 128, 256]
    
    def __call__(self):
        if self.model in ['resnet18', 'resnet50']:
            x_input = Input(self.input_shape)
            encoder = self.backbone(x_input)
            model = self.decoder(encoder)
        elif self.model == 'vgg66':
            model = vgg(self)
        else:
            model = unet(self)

        return model

    def _resnet18(self, x_input):
        x = Conv2D(self.filters[0], strides=1, kernel_size=(7, 7),
                padding='same', kernel_initializer='he_normal')(x_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

        # -------------Encoder--------------
        x = self._resnet_block(x, self.filters[0], dil_rate=self.dil_rate, name='enc_1.0')
        x = self._resnet_block(x, self.filters[0], dil_rate=self.dil_rate, name='enc_1')
        x = self._resnet_block(x, self.filters[1], dil_rate=self.dil_rate, name='enc_2.0', down_sample=True)
        x = self._resnet_block(x, self.filters[1], dil_rate=self.dil_rate, name='enc_2')
        x = self._resnet_block(x, self.filters[2], dil_rate=self.dil_rate, name='enc_3.0', down_sample=True)
        x = self._resnet_block(x, self.filters[2], dil_rate=self.dil_rate, name='enc_3')
        x = self._resnet_block(x, self.filters[3], dil_rate=self.dil_rate, name='enc_4.0', down_sample=True)
        x = self._resnet_block(x, self.filters[3], dil_rate=self.dil_rate, name='enc_4')

        return Model(x_input, x, name="resnet18")
    
    def _resnet50(self, x_input):
        # self.filters = [64, 256, 512, 1024, 2024]

        resnet50 = ResNet50(include_top  = False,
                            weights      = self.weights,
                            input_shape  = self.input_shape,
                            input_tensor = x_input)
        
        resnet50.get_layer("conv1_relu")._name       = 'enc_1'
        resnet50.get_layer("conv2_block3_out")._name = 'enc_2'
        resnet50.get_layer("conv3_block4_out")._name = 'enc_3'
        resnet50.get_layer("conv4_block6_out")._name = 'enc_4'

        x = resnet50.get_layer('enc_4').output

        return Model(x_input, x, name="resnet50")

    def backbone(self, x_input):
        backbone_models = {
            'resnet18': self._resnet18,
            'resnet50': self._resnet50
        }

        backbone = backbone_models.get(self.model, None)(x_input)

        if self.weights == 'imagenet':
            if self.model == 'resnet18':
                weights_path = get_file('resnet18_imagenet_1000_no_top.h5',
                                        WEIGHTS_PATH_NO_TOP_18,
                                        cache_subdir='models',
                                        md5_hash='318e3ac0cd98d51e917526c9f62f0b50')
                backbone.load_weights(weights_path, by_name=True)
                
            for layer in backbone.layers:
                layer.trainable = False
                if layer.name == self.layer_to_freeze:
                    break

        return backbone

    def decoder(self, backbone):
        skip_x = []
        skip_x.append(backbone.get_layer("enc_1").output)
        skip_x.append(backbone.get_layer("enc_2").output)
        skip_x.append(backbone.get_layer("enc_3").output)
        skip_x.append(backbone.get_layer("enc_4").output)

        # tf.print(skip_x[0].shape)
        # tf.print(skip_x[1].shape)
        # tf.print(skip_x[2].shape)
        # tf.print(skip_x[3].shape)

        # -------------Bottleneck-------------
        x = self._resblock(backbone.output, self.filters[4], 'bottleneck')
        # x = self._resnet_block(backbone.output, self.filters[4], dil_rate=self.dil_rate, name='bottleneck', down_sample=True)
        skip_x.append(x)
        
        # tf.print('\n', skip_x[4].shape, '\n')

        # -------------Decoder--------------
        for i in range(len(skip_x) - 2, -1, -1):
            xs = skip_x[i]
            # x = UpSampling2D()(x)
            # x = Conv2D(self.filters[i], 2, activation='relu', padding='same')(x)
            # x = Concatenate(axis=-1)([x, xs])
            # x = self._conv_block(x, self.filters[i], name=f'dec_{4-i}')
            x = self._upsample_concat(x, xs)
            x = self._resblock(x, self.filters[i], f'dec_{i+1}')

            # tf.print(x.shape)

        # --------------Output---------------
        activation = 'sigmoid' if self.num_class == 1 else 'softmax'
        output = Conv2D(self.num_class, (1, 1), activation=activation, padding="same", name='seg_mask')(x)

        return Model(backbone.input, output)

    # ---> blocchi convoluizonali <---
    def _resnet_block(self, x, filters: int, kernel_size=3, name='', init_scheme='he_normal', down_sample=False, dil_rate = 1):
        strides = [2, 1] if down_sample else [1, 1]
        res = x

        x = Conv2D(filters, strides=strides[0], kernel_size=kernel_size, 
                                padding='same', kernel_initializer=init_scheme)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters, strides=strides[1], kernel_size=kernel_size, 
                                padding='same', kernel_initializer=init_scheme,
                                dilation_rate=dil_rate)(x)
        x = BatchNormalization()(x)
        
        if down_sample:
            # perform down sampling using stride of 2, according to [1].
            res = Conv2D(filters, strides=2, kernel_size=(1, 1),
                                padding='same', kernel_initializer=init_scheme)(res)

        # if down sampling not enabled, add a shortcut directly
        x = Add()([x, res])
        out = Activation('relu', name = name)(x)

        return out

    def _upsample_concat(self, x, skip):
        # Use UpSampling2D layer for bilinear upsampling to match skip's spatial dimensions
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

        # Reduce the number of channels in skip to match x
        skip_channels = x.shape[-1]
        skip = Conv2D(skip_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(skip)
        skip = BatchNormalization()(skip)
        skip = Activation('relu')(skip)

        # Use another UpSampling2D layer to match the spatial dimensions of x
        skip = UpSampling2D(size=(2, 2), interpolation='bilinear')(skip)

        # Concatenate the upsampled tensor with the modified skip tensor
        merge = Concatenate()([x, skip])
        return merge

    def _resblock(self, x, f, name):
        x_copy = x

        x = Conv2D(f, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(f, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x_copy = Conv2D(f, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal')(x_copy)
        x_copy = BatchNormalization()(x_copy)

        x = Add()([x, x_copy])
        x = Activation('relu', name = name)(x)

        return x
    
    # ---> blocchi convoluizonali <---

def unet(setting):
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

    input_shape = setting.input_shape
    filters     = setting.filters
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

def vgg(setting):
    '''Build the unet model'''

    # -------------Encoder--------------
    base_VGG = VGG16(include_top = False,
                     weights     = setting.weights,
                     input_shape = setting.input_shape)

    conv1 = base_VGG.get_layer("block1_conv2").output
    conv2 = base_VGG.get_layer("block2_conv2").output
    conv3 = base_VGG.get_layer("block3_conv3").output
    conv4 = base_VGG.get_layer("block4_conv3").output

    base_VGG.get_layer("block1_conv2")._name = 'enc_1'
    base_VGG.get_layer("block2_conv2")._name = 'enc_2'
    base_VGG.get_layer("block3_conv3")._name = 'enc_3'
    base_VGG.get_layer("block4_conv3")._name = 'enc_4'

    if setting.weights is not None:
        for layer in base_VGG.layers:
            layer.trainable = False
            if layer.name == setting.layer_to_freeze:
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