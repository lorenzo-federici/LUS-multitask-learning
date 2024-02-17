import tensorflow as tf
from keras.utils import get_file
import platform
from tensorflow.keras.applications import ResNet50

from keras.layers import *
from keras.models import *
from keras.backend import *
from keras_cv.models import *

WEIGHTS_PATH_NO_TOP_18 = 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5'

def _resnet_block(x, filters: int, kernel_size=3, name='', init_scheme='he_normal', down_sample=False, dil_rate = 1):
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

def _convolution_block(x, filters, size, batchnorm, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def _residual_block(blockInput, batchnorm, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    if batchnorm:
        x = BatchNormalization()(x)
        blockInput = BatchNormalization()(blockInput)
    x = _convolution_block(x, num_filters, (3,3), batchnorm)
    x = _convolution_block(x, num_filters, (3,3), batchnorm, activation=False)
    x = Add()([x, blockInput])
    return x
   
def _upsample_concat(x, skip):
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

def _resblock(x, f, name):
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
    

class ResUnet():
    def __init__(self, input_shape= (224,224,3), model = 'resnet50pp', weights=None, layer_to_freeze=None, num_class=1, dropout=.0, dil_rate = 1):
        self.input_shape = input_shape
        self.model = model
        self.weights = weights
        self.layer_to_freeze = layer_to_freeze
        self.num_class = num_class
        self.dropout = dropout
        self.dil_rate = dil_rate

        self.start_neurons = 8

    def __call__(self):
        # _SO = platform.system()
        # if _SO == "Darwin":
        #     self.start_neurons = 8 # [16, 32, 64, 128, 256]
        # else:
        #     self.start_neurons = 32 # [32, 64, 128, 256, 512]

        self.start_neurons = 4 # [16, 32, 64, 128, 256]

        # model = self.define_model()
        x_input = Input(self.input_shape)
        encoder = self.backbone(x_input)
        if 'pp' in self.model:
            model = self.decoder_pp(encoder)
        else:
            model = self.decoder_std(encoder)

        return model
    
    def backbone(self, x_input):
        backbone_models = {
            'resnet18': self._resnet18,
            'resnet50': self._resnet50,
            'resnet18pp': self._resnet18,
            'resnet50pp': self._resnet50
        }

        backbone = backbone_models.get(self.model, None)(x_input)

        if self.weights == 'imagenet':
            if 'resnet18' in self.model:
                weights_path = get_file('resnet18_imagenet_1000_no_top.h5',
                                        WEIGHTS_PATH_NO_TOP_18,
                                        cache_subdir='models',
                                        md5_hash='318e3ac0cd98d51e917526c9f62f0b50')
                backbone.load_weights(weights_path, by_name=True)
                
            # for layer in backbone.layers:
            #     layer.trainable = False
            #     if layer.name == self.layer_to_freeze:
            #         break

        return backbone
    
    def _resnet18(self, x_input):
        # start_neurons = self.start_neurons * 2
        start_neurons = self.start_neurons
        
        x = Conv2D(start_neurons * 2, strides=1, kernel_size=(7, 7),
                padding='same', kernel_initializer='he_normal')(x_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

        # -------------Encoder--------------
        x = _resnet_block(x, start_neurons * 2, dil_rate=self.dil_rate, name='enc_1.0')
        x = _resnet_block(x, start_neurons * 2, dil_rate=self.dil_rate, name='enc_1')
        x = _resnet_block(x, start_neurons * 4, dil_rate=self.dil_rate, name='enc_2.0', down_sample=True)
        x = _resnet_block(x, start_neurons * 4, dil_rate=self.dil_rate, name='enc_2')
        x = _resnet_block(x, start_neurons * 8, dil_rate=self.dil_rate, name='enc_3.0', down_sample=True)
        x = _resnet_block(x, start_neurons * 8, dil_rate=self.dil_rate, name='enc_3')
        x = _resnet_block(x, start_neurons * 16, dil_rate=self.dil_rate, name='enc_4.0', down_sample=True)
        x = _resnet_block(x, start_neurons * 16, dil_rate=self.dil_rate, name='enc_4')

        return Model(x_input, x, name="resnet18")
    
    def _resnet50(self, x_input):
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
    
    def decoder_std(self, backbone):
        skip_x = []
        filters = [self.start_neurons*2, self.start_neurons*4, self.start_neurons*8, self.start_neurons*16, self.start_neurons*32]
        skip_x.append(backbone.get_layer("enc_1").output)
        skip_x.append(backbone.get_layer("enc_2").output)
        skip_x.append(backbone.get_layer("enc_3").output)
        skip_x.append(backbone.get_layer("enc_4").output)

        # -------------Bottleneck-------------
        x = _resblock(backbone.output, filters[4], 'bottleneck')
        skip_x.append(x)

        # -------------Decoder--------------
        for i in range(len(skip_x) - 2, -1, -1):
            xs = skip_x[i]
            x = _upsample_concat(x, xs)
            x = _resblock(x, filters[i], f'dec_{i+1}')

        # --------------Output---------------
        activation = 'sigmoid' if self.num_class == 1 else 'softmax'
        output = Conv2D(self.num_class, (1, 1), activation=activation, padding="same", name='seg_mask')(x)

        return Model(backbone.input, output, name = self.model)

    def decoder_pp(self, backbone):
        start_neurons = self.start_neurons
        dropout_rate = self.dropout
        batchnorm = True

        conv4 = backbone.get_layer("enc_4").output # conv4_block6_out
        conv4 = LeakyReLU(alpha=0.1)(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4) 
        pool4 = Dropout(dropout_rate)(pool4)

        # Middle
        convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same",name='conv_middle')(pool4)
        convm = _residual_block(convm, batchnorm, start_neurons * 32)
        convm = _residual_block(convm, batchnorm, start_neurons * 32)
        convm = LeakyReLU(alpha=0.1, name = 'bottleneck')(convm)
        
        deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
        deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
        deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
        deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(dropout_rate)(uconv4) 
        
        uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = _residual_block(uconv4, batchnorm, start_neurons * 16)
        uconv4 = LeakyReLU(alpha=0.1, name = 'dec_4')(uconv4) 
        
        deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
        deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
        deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)
        conv3 = backbone.get_layer("enc_3").output # conv3_block4_out
        uconv3 = concatenate([deconv3,deconv4_up1, conv3])    
        uconv3 = Dropout(dropout_rate)(uconv3)
        
        uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = _residual_block(uconv3, batchnorm, start_neurons * 8)
        uconv3 = LeakyReLU(alpha=0.1)(uconv3)

        deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3) 
        deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2) 
        conv2 = backbone.get_layer("enc_2").output #conv2_block3_out
        uconv2 = concatenate([deconv2,deconv3_up1,deconv4_up2, conv2])
            
        uconv2 = Dropout(dropout_rate)(uconv2)
        uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = _residual_block(uconv2, batchnorm, start_neurons * 4)
        uconv2 = LeakyReLU(alpha=0.1)(uconv2)
        
        deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2) 
        conv1 = backbone.get_layer("enc_1").output #conv1_relu
        uconv1 = concatenate([deconv1,deconv2_up1,deconv3_up2,deconv4_up3, conv1])
        
        uconv1 = Dropout(dropout_rate)(uconv1)
        uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)  
        uconv1 = _residual_block(uconv1, batchnorm, start_neurons * 2)
        uconv1 = LeakyReLU(alpha=0.1)(uconv1)
        
        uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)  
        uconv0 = Dropout(dropout_rate)(uconv0)
        uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
        uconv0 = _residual_block(uconv0, batchnorm, start_neurons * 1)
        uconv0 = LeakyReLU(alpha=0.1,  name = 'dec_1')(uconv0)
        
        uconv0 = Dropout(dropout_rate)(uconv0)

        # --------------Output---------------
        activation = 'sigmoid' if self.num_class == 1 else 'softmax'
        output = Conv2D(self.num_class, (1, 1), activation=activation, padding="same", name='seg_mask')(uconv0)

        return Model(backbone.input, output, name = self.model)