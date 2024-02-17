from typing import Any
import keras
import tensorflow as tf
from keras.layers import *
from keras.models import *

from tensorflow.keras.applications import ResNet50

from keras.utils import get_file

WEIGHTS_PATH_NO_TOP_18 = 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5'

class ResNet():
    def __init__(self, input_shape= (224,224,3), model = 'resnet18', weights=None, layer_to_freeze=None, num_class=4, dropout=.0, dil_rate = 1):
        self.input_shape     = input_shape
        self.model           = model
        self.weights         = None if weights == 'none' else weights
        self.layer_to_freeze = None if layer_to_freeze == 'none' else layer_to_freeze
        self.num_class       = num_class
        self.dropout         = dropout
        self.dil_rate        = dil_rate

        self.start_neurons   = 4
    
    def __call__(self):
        if self.model == 'resnet18':
            model = self.resnet18()
        else:
            model = self.resnet50()
        return model
    
    def resnet18(self):
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

        input = Input(shape=self.input_shape)

        start_neurons = self.start_neurons * 2

        x = Conv2D(start_neurons * 2, strides=2, kernel_size=(7, 7), 
                    padding='same', kernel_initializer='he_normal')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

        x = _resnet_block(x, start_neurons * 2, dil_rate=self.dil_rate, name='enc_1.0')
        x = _resnet_block(x, start_neurons * 2, dil_rate=self.dil_rate, name='enc_1')
        x = _resnet_block(x, start_neurons * 4, dil_rate=self.dil_rate, name='enc_2.0', down_sample=True)
        x = _resnet_block(x, start_neurons * 4, dil_rate=self.dil_rate, name='enc_2')
        x = _resnet_block(x, start_neurons * 8, dil_rate=self.dil_rate, name='enc_3.0', down_sample=True)
        x = _resnet_block(x, start_neurons * 8, dil_rate=self.dil_rate, name='enc_3')
        x = _resnet_block(x, start_neurons * 16, dil_rate=self.dil_rate, name='enc_4.0', down_sample=True)
        x = _resnet_block(x, start_neurons * 16, dil_rate=self.dil_rate, name='enc_4')

        if self.dropout > .0:
            x = Dropout(rate=self.dropout)(x)

        if not self.weights == 'imagenet':
            x = self._classification_head(x, self.num_class, name='')
        
        model = Model(input, x, name="resnet18")

        # load weights
        if self.weights == 'imagenet':
            weights_path = get_file('resnet18_imagenet_1000_no_top.h5',
                                    WEIGHTS_PATH_NO_TOP_18,
                                    cache_subdir='models',
                                    md5_hash='318e3ac0cd98d51e917526c9f62f0b50')
            model.load_weights(weights_path, by_name=True)
            self._fine_tuning(model, self.layer_to_freeze)
            x = self._classification_head(model.output, self.num_class, name='_imagenet')

            model  = Model(inputs=[model.input], outputs=x)

        return model

    def resnet50(self):
        base_model = ResNet50(weights=self.weights, include_top=False, input_shape = self.input_shape)
        name = ''

        if self.weights == 'imagenet':
            if self.layer_to_freeze == 'enc_2':
                self.layer_to_freeze = base_model.layers[35].name
            self._fine_tuning(base_model, self.layer_to_freeze)
            name = '_imagenet'
        
        x = self._classification_head(base_model.output, self.num_class, name=name)
        
        model = Model(base_model.input, x, name="resnet50")
        return model

    def _classification_head(self, x, num_class, name=''):
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(num_class, activation = 'softmax', name = 'cls_output' + name)(x)
        return x

    def _fine_tuning(self, model, stop_layer_name):
        for layer in model.layers:
            layer.trainable = False
            if layer.name == stop_layer_name:
                break