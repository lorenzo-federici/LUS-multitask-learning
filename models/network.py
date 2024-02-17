import numpy as np
import keras
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Flatten, Dropout

# from models.resnet18 import resnet18
from models.resnet import ResNet
from models.resUnet import ResUnet
from models.unet import *
from models.unetpp import *

class Network:
    def __init__(self, exp_config, num_classes_seg:int = 1, num_classes_cls:int = 4, input_size:int = 224):
        self.input_size  = (input_size, input_size, 3)
        self.n_class_cls = 4
        self.n_class_seg = 1
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
        
        if self.task != 'classification' and self.backbone != 'unet':
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
            seg_model = ResUnet(self.input_size, 
                                model = self.backbone, 
                                weights=self.weights, 
                                layer_to_freeze=self.layer_to_freeze, 
                                num_class=self.n_class_seg, 
                                dropout=self.dropout, 
                                dil_rate = self.dil_rate)()
            cls_model = self.classification_model(seg_model)

            output = [cls_model.output, seg_model.output]  # Combine outputs.
            model  = Model(inputs=[seg_model.input], outputs=output, name=f'multitask_{self.model_name}')
        elif self.task == 'segmentation':
            model = ResUnet(self.input_size, 
                            model = self.backbone, 
                            weights=self.weights, 
                            layer_to_freeze=self.layer_to_freeze, 
                            num_class=self.n_class_seg, 
                            dropout=self.dropout, 
                            dil_rate = self.dil_rate)()
        elif self.task == 'classification':
            model = ResNet(self.input_size, 
                           model = self.backbone, 
                           weights=self.weights, 
                           layer_to_freeze=self.layer_to_freeze, 
                           num_class=self.n_class_cls, 
                           dropout=self.dropout, 
                           dil_rate = self.dil_rate)()     
        
        return model
    
    def classification_model(self, base_model):
        '''Classificator model'''
        conv4_layer       = base_model.get_layer("enc_4")
        bottleneck_layer  = base_model.get_layer("bottleneck")
        dec4_layer        = base_model.get_layer("dec_4")
        
        conv4      = conv4_layer.output
        bottleneck = bottleneck_layer.output
        dec4       = dec4_layer.output
        
        xcon = GlobalAveragePooling2D(keepdims = False)(conv4)
        xbot = GlobalAveragePooling2D(keepdims = False)(bottleneck)
        xdec = GlobalAveragePooling2D(keepdims = False)(dec4)

        x = tf.concat([xcon, xbot, xdec], axis=-1)

        x = Dense(256, activation='relu')(x)
        if self.dropout > .0:
            x = Dropout(rate=self.dropout)(x)

        # x = Dense(128, activation='relu', name='last')(x)
        # if self.dropout > .0:
        #     x = Dropout(rate=self.dropout)(x)
            
        output = Dense(self.n_class_cls, activation='softmax', name='cls_label')(x)

        return Model(inputs=base_model.input, outputs=output, name='classificator')