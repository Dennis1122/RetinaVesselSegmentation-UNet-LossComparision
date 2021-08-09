import tensorflow as tf
import tensorflow.keras.backend as K


def iou(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred) + 1.0
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.0
    return intersection / union


def dice(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred) + 1.0
    dice_score = (2. * intersection + 1.0) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.0)
    return dice_score


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)