import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from utils.dice_helpers_tf import soft_cldice_loss
import tensorflow.keras.backend as K


# Dice Losses
def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = K.sum(y_true * y_pred)
    dice_score = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    dice_loss = 1. - dice_score
    return dice_loss


def focal_loss(y_true, y_pred, gamma=2., alpha=.5):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss


cldice_loss = soft_cldice_loss(k=10, data_format="channels_last")


# combine dice + cldice
def combined_cldice_loss(y_true, y_pred, k=10, alpha=0.5):
    data_format = "channels_last"
    return (alpha * dice_loss(data_format=data_format)(y_true, y_pred) +
            (1 - alpha) * soft_cldice_loss(k, data_format=data_format)(y_true, y_pred))


def tversky(y_true, y_pred, smooth=1, alpha=0.6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred, gamma=0.75):
    pt_1 = tversky(y_true, y_pred)
    return K.pow((1 - pt_1), gamma)


# combine clDice + tversky
def tversky_cldice_loss(y_true, y_pred, k=10, alpha=0.5):
    data_format = "channels_last"
    return (alpha * tversky_loss(y_true, y_pred) +
            (1 - alpha) * soft_cldice_loss(k, data_format=data_format)(y_true, y_pred))


# combine clDice + focal tversky
def focal_tversky_cldice_loss(y_true, y_pred, k=10, alpha=0.5):
    data_format = "channels_last"
    return (alpha * focal_tversky(y_true, y_pred) +
            (1 - alpha) * soft_cldice_loss(k, data_format=data_format)(y_true, y_pred))