import tensorflow as tf


def res_block(scope, inputs, filters, strides, use_bias, bn_func):
    bn_dict = dict()
    shortcut = inputs
    if strides != 1:
        shortcut = tf.layers.conv2d(inputs, filters, 1, strides, padding='same', use_bias=use_bias)
        shortcut, normed0 = bn_func(scope + '/shortcut', shortcut)
        bn_dict.update({'shortcut': shortcut})

    inputs, normed1 = bn_func(scope + '/conv_0', inputs)
    bn_dict.update({'conv_0': normed1})
    inputs = tf.nn.relu(inputs)

    inputs = tf.layers.conv2d(inputs, filters, 3, strides, padding='same', use_bias=use_bias)
    inputs, normed2 = bn_func(scope + '/conv_1', inputs)
    bn_dict.update({'conv_1': normed2})
    inputs = tf.nn.relu(inputs)

    inputs = tf.layers.conv2d(inputs, filters, 3, padding='same', use_bias=use_bias)
    inputs += shortcut

    return tf.nn.relu(inputs), bn_dict
