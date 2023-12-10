import numpy as np
import keras
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Flatten, Dropout

# from models.resnet18 import resnet18
from models.resnet import ResNet
from models.unet import UNet
from models.resUnet import *
from models.resnet50 import *


class Network:
    def __init__(self, exp_config, num_classes_seg:int = 1, num_classes_cls:int = 4, input_size:int = 224):
        self.input_size  = (input_size, input_size, 3)
        self.n_class_cls = num_classes_cls
        self.n_class_seg = num_classes_seg
        self.exp_config  = exp_config
        self.filters     = [16, 32, 64, 128, 256] #, 512, 1024]

        self.task     = exp_config.get('task', None)
        self.backbone = exp_config.get('backbone', None)
        self.dropout  = exp_config.get('dropout', .0)
        self.weights  = exp_config.get('weights', None)
        self.layer_to_freeze = exp_config.get('layer_to_freeze', None)
        self.dil_rate = exp_config.get('dil_rate', 1)
        
        self.model_name = self.get_model_name()

    def get_model_name(self):
        '''Create model name'''

        name = self.backbone
        
        if self.task is not 'classification':
            name = name + "_UNet"
        
        return name
    
    def build_model(self):
        print('>> BUILDING MODEL:')
        print(f'\t• Task --> {self.task}')
        print(f'\t• Model --> {self.model_name}')
        print(f'\t• Batch size --> {self.exp_config["batch_size"]}')
        print(f'\t• Optimizer --> {self.exp_config["optimizer"]} lr:{self.exp_config["lr"]}')
        print(f'\t• Epoch --> {self.exp_config["epoch"]}')
        print(f'\t• Model --> {self.exp_config["augmentation"]}')
        print(f'\t• Dropout --> {self.dropout}')

        if not self.weights == None:
            print('\t• Trasfer learning Active from IMAGENET')
            print(f'\t• Freeze --> {self.layer_to_freeze}')

        if self.task == 'multitask':
            seg_model = self.segmentation_model()
            cls_model = self.classification_model(seg_model)

            output = [cls_model.output, seg_model.output]  # Combine outputs.
            model  = Model(inputs=[seg_model.input], outputs=output, name=f'multitask_{self.model_name}')
        elif self.task == 'segmentation':
            # model = resnet50()
            # model = UNet()
            model = ResUNet(self.input_size, model = self.backbone, weights=self.weights, layer_to_freeze=self.layer_to_freeze, num_class=self.n_class_seg, dropout=self.dropout, dil_rate = self.dil_rate)()
        elif self.task == 'classification':
            model = ResNet(self.input_size, model = self.backbone, weights=self.weights, layer_to_freeze=self.layer_to_freeze, num_class=self.n_class_cls, dropout=self.dropout, dil_rate = self.dil_rate)()        
        
        return model
    
    def _resnet18_backbone(self, x):
        '''ResNet18 backbone''' 
        def _residual_block(x, filter, name, kernel_size=3, stride=1):
            '''Residual Block for ResUnet/ResNet'''
            shortcut = x

            x = Conv2D(filter, kernel_size, strides=stride, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(filter, kernel_size, padding='same')(x)
            x = BatchNormalization()(x)

            if stride != 1 or shortcut.shape[-1] != filter:
                shortcut = Conv2D(filter, 1, strides=stride)(shortcut)
                shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu', name = name)(x)
            return x

        filters = self.filters

        x = Conv2D(filters[0], 7, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

        # -------------Encoder--------------
        num_blocks = [2, 2, 2, 2, 1]  # Residual Block number for each encoder stages
        skip_x = []

        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                stride = 2 if i > 0 and j == 0 else 1
                if i == len(filters) - 1:
                    name = 'bottleneck'
                else:
                    name = f'enc_{i+1}.0' if j == 0 else f'enc_{i+1}'
                x = _residual_block(x, filters[i], name, stride=stride)
            skip_x.append(x)

        return skip_x

    def segmentation_model(self):
        '''Segmentation Model'''
        def _conv_block(x_input, filters, name):
            '''Convolution Block'''
            x = Conv2D(filters, (3, 3), padding="same")(x_input)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)

            if self.backbone == 'ResNet':
                res = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x_input)
                x   = Add()([res, x])

            x = Activation("relu", name=name)(x)
            return x
        
        x_inputs = Input(self.input_size)
        filters  = self.filters
        skip_x   = []

        # -------------Encoder--------------
        if self.backbone == 'ResNet':
            # Resnet 18 Backbone
            skip_x = self._resnet18_backbone(x_inputs)
        else:
            for i, filter in enumerate(filters):
                if i == len(filters) - 1:
                    name = 'bottleneck'
                else:
                    name = f'enc_{i+1}'
                x = _conv_block(x, filter, name)
                skip_x.append(x)
                if i < len(filters) - 1:
                    # Bridge doesn't want maxpooling
                    x = MaxPooling2D((2, 2))(x)

        # -------------bottleneck-----------
        x = skip_x[-1]

        # -------------Decoder--------------
        for i in range(len(skip_x) - 2, -1, -1):
            x = UpSampling2D()(x)
            if self.backbone == 'ResNet': 
                x = Conv2D(filters[i], 2, activation='relu', padding='same')(x)
            x = Concatenate(axis=-1)([x, skip_x[i]])
            x = _conv_block(x, filters[i], f'dec_{4-i}')

        # --------------Output---------------
        activation = 'sigmoid' if self.n_class_seg == 1 else 'softmax'
        output = Conv2D(self.n_class_seg, (1, 1), activation=activation, padding="same", name='seg_mask')(x)

        return Model(inputs=x_inputs, outputs=output, name=self.model_name)
    
    def classification_model(self, base_model):
        '''Classificator model'''
        conv4_layer       = base_model.get_layer("enc_4")
        bottleneck_layer  = base_model.get_layer("bottleneck")
        dec4_layer        = base_model.get_layer("dec_4")
        
        conv4      = conv4_layer.output
        bottleneck = bottleneck_layer.output
        dec4       = dec4_layer.output
        
        # tf.print(conv4.shape)
        # tf.print(bottleneck.shape)
        # tf.print(dec4.shape)

        xcon = GlobalAveragePooling2D(keepdims = False)(conv4)
        xbot = GlobalAveragePooling2D(keepdims = False)(bottleneck)
        xdec = GlobalAveragePooling2D(keepdims = False)(dec4)

        x = tf.concat([xcon, xbot, xdec], axis=-1)

        x = Dense(256, activation='relu')(x)
        if self.dropout > .0:
            x = Dropout(rate=self.dropout)(x)
        x = Dense(128, activation='relu')(x)
        if self.dropout > .0:
            x = Dropout(rate=self.dropout)(x)
            
        output = Dense(self.n_class_cls, activation='softmax', name='cls_label')(x)

        return Model(inputs=base_model.input, outputs=output, name='classificator')