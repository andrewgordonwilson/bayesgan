import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from dcgan_utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=False,
                                            center=False,
                                            is_training=train,
                                            scope=self.name)

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", w=None, biases=None):
    with tf.variable_scope(name):
        if w is None:
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        if biases is None:
            biases = tf.get_variable(
                'biases',
                [output_dim],
                initializer=tf.constant_initializer(0.0))
            
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def wn_conv2d(x, num_output_filters, filter_size, stride,
              V, g, b, pad="SAME", name="wn_conv2d"):

    with tf.variable_scope(name):
        #W = tf.reshape(g, [1,1,1,num_output_filters])*tf.nn.l2_normalize(V,[0,1,2])
        W = V
        conv = tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1]+stride+[1], padding=pad), b)
        out = lrelu(conv)
        return out



def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, w=None, biases=None):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        if w is None:
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        if biases is None:
            biases = tf.get_variable('biases',
                                     [output_shape[-1]],
                                     initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None,
           stddev=0.02, bias_start=0.0, with_w=False, matrix=None, bias=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        if matrix is None:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
        if bias is None:
            bias = tf.get_variable("bias", [output_size],
                                   initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
