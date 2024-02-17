import tensorflow as tf
from tensorflow.keras import backend as K

# Segmentation -----------------------------------------------------------------------------------
def dice_coef(y_true, y_pred):
    '''Evaluation metrics: dice coefficient '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    smooth = 1
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth = 1.):
    '''Evaluation metrics: iou'''
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true) + K.sum(y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def tversky(y_true, y_pred, smooth = 1e-6):
    '''Evaluation metrics: tversky'''
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
def F1_score(y_true, y_pred):
    # Convert predictions to one-hot encoded vectors
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=K.int_shape(y_pred)[-1])

    # True Positives, False Positives, and False Negatives
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    # Precision and Recall for each class
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    # F1 Score for each class
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    # Replace NaNs with zeros and calculate the average
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
# ------------------------------------------------------------------------------------------------
