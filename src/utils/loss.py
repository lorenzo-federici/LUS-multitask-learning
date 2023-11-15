import tensorflow as tf
from tensorflow.keras import backend as K
from keras.losses import categorical_crossentropy

from utils.metrics import *


# Segmentation -----------------------------------------------------------------------------------
def dice_coef_loss(y_true, y_pred):
    '''Dice coefficient loss'''
    return 1-dice_coef(y_true, y_pred)

def iou_loss(y_true, y_pred, smooth = 1.):
    '''IoU loss'''
    return 1-iou(y_true, y_pred)

@tf.function
def focal_tversky(y_true, y_pred):
    '''Focal tversky loss'''
    alpha = 0.7
    beta  = 0.3
    smooth = 1e-6

    # Flatten the inputs
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)

    # Calculate true positives, false positives, and false negatives
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    # Calculate Tversky score
    tversky_score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    # Calculate focal Tversky loss
    focal_tversky_loss = tf.pow(1 - tversky_score, 0.5)

    return focal_tversky_loss
# ------------------------------------------------------------------------------------------------

# Classification ---------------------------------------------------------------------------------
def weighted_categorical_crossentropy(weights):
    '''
        weighted version of keras.objectives.categorical_crossentropy
        weights: numpy array of shape (C,) where C is the number of classes
    '''
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
# ------------------------------------------------------------------------------------------------