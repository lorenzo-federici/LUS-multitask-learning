import tensorflow as tf

from keras.layers import *
from keras.models import *
from keras.backend import *
from keras_cv.models import *

def UNet(input_size = (224, 224, 3)):
    '''Build the unet model'''

    def conv_block(x, num_filters):
        '''Convolution Block'''
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    inputs = Input(input_size)

    # num_filters = [64, 128, 256, 512, 1024]
    # num_filters = [32, 64, 128, 256, 512]
    # num_filters = [16, 32, 64, 128, 256]
    num_filters = [8, 16, 32, 64, 128]

    skip_x = []
    x = inputs

    # -------------Encoder--------------
    for f in num_filters[:-1]:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)
        
    tf.print(skip_x[0].shape)
    tf.print(skip_x[1].shape)
    tf.print(skip_x[2].shape)
    tf.print(skip_x[3].shape)

    # -------------Bridge--------------
    x = conv_block(x, num_filters[-1])
    tf.print(x.shape)

    num_filters.reverse()
    skip_x.reverse()

    tf.print('\n\n')
    # -------------Decoder--------------
    for i, f in enumerate(num_filters[:-1]):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x,xs])
        x = conv_block(x, f)
        tf.print(x.shape)

    # --------------Output---------------
    output = Conv2D(1, (1, 1), padding="same")(x)
    output = Activation("sigmoid")(output)

    return Model(inputs, output)