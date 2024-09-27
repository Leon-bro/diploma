from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
# Define IoU Metric (Intersection over Union)
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_metric(y_true, y_pred, smooth=1e-6, gama=2):
    y_true, y_pred = tf.cast(
        y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * \
                tf.reduce_sum(tf.multiply(y_pred, y_true)) + smooth
    denominator = tf.reduce_sum(
        y_pred ** gama) + tf.reduce_sum(y_true ** gama) + smooth
    result = tf.divide(nominator, denominator)
    return result