import numpy as np
from keras import backend as K
import tensorflow as tf


def uu_loss(loss_type, theta1, theta2, prior, mode, surro_flag):
    def sigmoid_loss(x):
        condition = tf.less_equal(x, 0)
        loss_val = tf.where(condition,
                            1. / (1. + tf.exp(x)),
                            tf.exp(-x) / (1. + tf.exp(-x)))
        return loss_val

    def log_loss(x):
        return tf.nn.softplus(-x)

    def make_loss(loss_name, y):
        if loss_name == 'sigmoid':
            return sigmoid_loss(y)
        elif loss_name == 'logistic':
            return log_loss(y)
        else:
            ValueError("Loss name {} is unknown.".format(loss_name))

    def loss(loss_pos, loss_neg, y_true, y_pred):
        assert y_pred.shape[1] == 1
        assert y_pred.shape.ndims == 2

        positive = tf.cast(tf.equal(tf.ones_like(y_true), y_true), tf.float64)
        negative = tf.cast(tf.equal(-tf.ones_like(y_true), y_true), tf.float64)
        num_pos = tf.maximum(tf.cast(1, tf.float64), tf.reduce_sum(positive))
        num_neg = tf.maximum(tf.cast(1, tf.float64), tf.reduce_sum(negative))

        if mode == "UU":
            weight1 = tf.cast((prior * (1. - theta2)) / (theta1 - theta2), tf.float64)
            weight2 = tf.cast((theta2 * (1. - prior)) / (theta1 - theta2), tf.float64)
            weight3 = tf.cast((prior * (1. - theta1)) / (theta1 - theta2), tf.float64)
            weight4 = tf.cast((theta1 * (1. - prior)) / (theta1 - theta2), tf.float64)
            pos_risk_UU = tf.reduce_sum((weight1 * loss_pos - weight2 * loss_neg) * positive) / num_pos
            neg_risk_UU = tf.reduce_sum((-weight3 * loss_pos + weight4 * loss_neg) * negative) / num_neg
            objective_UU = tf.cast(pos_risk_UU + neg_risk_UU, tf.float32)
            return objective_UU

        elif mode == "UU_simp":
            weight1_simp = tf.cast((prior + (1. - 2. * prior) * theta2) / (theta1 - theta2), tf.float64)
            weight2_simp = tf.cast((prior + (1. - 2. * prior) * theta1) / (theta1 - theta2), tf.float64)
            pos_risk_UU_simp = tf.reduce_sum(weight1_simp * loss_pos * positive) / num_pos
            neg_risk_UU_simp = tf.reduce_sum(weight2_simp * loss_neg * negative) / num_neg
            const_simp = tf.cast((theta2 * (1. - prior) + (1. - theta1) * prior) / (theta1 - theta2), tf.float64)
            objective_UU_simp = tf.cast(pos_risk_UU_simp + neg_risk_UU_simp - const_simp, tf.float32)
            return objective_UU_simp

        elif mode == "PN" or mode == "small_PN":
            pos_risk_PN = tf.reduce_sum(
                tf.cast(prior, tf.float64) * tf.cast(positive, tf.float64) / num_pos * loss_pos)
            neg_risk_PN = tf.reduce_sum(
                tf.cast((1. - prior), tf.float64) * tf.cast(negative, tf.float64) / num_neg * loss_neg)
            objective_PN = tf.cast(pos_risk_PN + neg_risk_PN, tf.float32)
            return objective_PN

        else:
            raise ValueError("Test mode is unknown.")

    def surrogate_loss(y_true, y_pred):
        loss_pos = make_loss(loss_type, tf.cast(y_pred, tf.float64))
        loss_neg = make_loss(loss_type, tf.cast(-1. * y_pred, tf.float64))
        return loss(loss_pos, loss_neg, y_true, y_pred)

    def zeroone_loss(y_true, y_pred):
        loss_pos = tf.cast(tf.not_equal(tf.sign(y_pred), tf.ones_like(y_true)), tf.float64)
        loss_neg = tf.cast(tf.not_equal(tf.sign(y_pred), -tf.ones_like(y_true)), tf.float64)
        return loss(loss_pos, loss_neg, y_true, y_pred)

    if surro_flag:
        return surrogate_loss
    else:
        return zeroone_loss