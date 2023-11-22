import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *

from keras.utils import get_file

WEIGHTS_PATH_NO_TOP = 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5'

def resnet18(input_shape= (224,224,3), weights=None, layer_to_freeze=None, num_class=4, dropout=.0):
    def _resnet_block(x, filters: int, kernel_size=3, name='', init_scheme='he_normal', down_sample=False):
        strides = [2, 1] if down_sample else [1, 1]
        res = x

        x = Conv2D(filters, strides=strides[0], kernel_size=kernel_size, 
                                padding='same', kernel_initializer=init_scheme)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, strides=strides[1], kernel_size=kernel_size, 
                                padding='same', kernel_initializer=init_scheme,
                                dilation_rate=2)(x)
        x = BatchNormalization()(x)
        
        if down_sample:
            # perform down sampling using stride of 2, according to [1].
            res = Conv2D(filters, strides=2, kernel_size=(1, 1),
                                padding='same', kernel_initializer=init_scheme)(res)

        # if down sampling not enabled, add a shortcut directly
        x = Add()([x, res])
        out = Activation('relu', name = name)(x)

        return out

    input = Input(shape=input_shape)

    x = Conv2D( 64, strides=2, kernel_size=(7, 7), 
                padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = _resnet_block(x, 64, name='enc_1.0')
    x = _resnet_block(x, 64, name='enc_1')
    x = _resnet_block(x, 128, name='enc_2.0', down_sample=True)
    x = _resnet_block(x, 128, name='enc_2')
    x = _resnet_block(x, 256, name='enc_3.0', down_sample=True)
    x = _resnet_block(x, 256, name='enc_3')
    x = _resnet_block(x, 512, name='enc_4.0', down_sample=True)
    x = _resnet_block(x, 512, name='enc_4')

    if dropout > .0:
        x = Dropout(rate=dropout)(x)

    # x = attention_block(x)

    if not weights == 'imagenet':
        x = _classification_head(x, num_class, name='')
    
    model = Model(input, x, name="resnet18")

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('resnet18_imagenet_1000_no_top.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='318e3ac0cd98d51e917526c9f62f0b50')
        model.load_weights(weights_path, by_name=True)
        _fine_tuning(model, layer_to_freeze)
        x = _classification_head(model.output, num_class, name='_imagenet')

        model  = Model(inputs=[model.input], outputs=x)

    return model

def _classification_head(x, num_class, name=''):
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(num_class, activation = 'softmax', name = 'cls_output' + name)(x)
    return x

def _fine_tuning(model, stop_layer_name):
    for layer in model.layers:
        layer.trainable = False
        if layer.name == stop_layer_name:
            break

def attention_block(x):
    # Attention network
    # a_map = Conv2D(1024, 1, strides=(1, 1), padding="same", activation='relu')(conv_base.output)
    # a_map = Conv2D(512, 1, strides=(1, 1), padding="same", activation='relu')(x)
    a_map = Conv2D(128, 1, strides=(1, 1), padding="same", activation='relu')(x)
    
    #a_map = Conv2D(64, 1, strides=(1, 1), padding="same", activation='relu')(a_map)
    a_map = Conv2D(1, 1, strides=(1, 1), padding="same", activation='relu')(a_map)
    
    #a_map = Conv2D(1024, 1, strides=(1, 1), padding="same", activation='relu')(a_map)
    # a_map = Conv2D(2048, 1, strides=(1, 1), padding="same", activation='sigmoid')(a_map)
    a_map = Conv2D(512, 1, strides=(1, 1), padding="same", activation='sigmoid')(a_map)
    res = Multiply()([x, a_map])
    return res


