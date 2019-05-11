from functools import partial
import tensorflow as tf


def none_batch_norm(scope, inputs, *args):
    return inputs, 0.


def batch_norm(scope, inputs, training_ph, *args):
    eps = 1e-8
    ndims = inputs.get_shape().ndims
    compress_axes = [i for i in range(ndims - 1)]
    shape = inputs.get_shape().as_list()[-1]
    with tf.variable_scope('{}/bn'.format(scope)):
        gamma = tf.get_variable('gamma', shape, tf.float32, tf.ones_initializer(), trainable=True)
        beta = tf.get_variable('beta', shape, tf.float32, tf.zeros_initializer(), trainable=True)
        mean = tf.get_variable('mean', shape, tf.float32, tf.zeros_initializer(), trainable=False)
        variance = tf.get_variable('variance', shape, tf.float32, tf.ones_initializer(), trainable=False)

    def train():
        decay = 0.999
        batch_mean, batch_variance = tf.nn.moments(inputs, compress_axes)
        update_mean_op = tf.assign(mean, decay * mean + (1 - decay) * batch_mean)
        update_variance_op = tf.assign(variance, decay * variance + (1 - decay) * batch_variance)
        with tf.control_dependencies([update_mean_op, update_variance_op]):
            normed = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, variance_epsilon=eps)
            return normed

    def test():
        normed = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon=eps)
        return normed

    normed = tf.cond(training_ph, train, test)
    return normed, 0.


def rigid_batch_norm(scope, inputs, training_ph, bound, *args):
    eps = 1e-8
    ndims = inputs.get_shape().ndims
    compress_axes = [i for i in range(ndims - 1)]
    shape = inputs.get_shape().as_list()[-1]
    with tf.variable_scope('{}/bn'.format(scope)):
        gamma = tf.get_variable('gamma', shape, tf.float32, tf.ones_initializer(), trainable=True)
        beta = tf.get_variable('beta', shape, tf.float32, tf.zeros_initializer(), trainable=True)
        mean = tf.get_variable('mean', shape, tf.float32, tf.zeros_initializer(), trainable=False)
        variance = tf.get_variable('variance', shape, tf.float32, tf.ones_initializer(), trainable=False)

    def train():
        decay = 0.999
        batch_mean, batch_variance = tf.nn.moments(inputs, compress_axes)
        pre_normalized = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, 0., 1., variance_epsilon=eps)

        non_omitted_bool = tf.logical_and(pre_normalized < bound, pre_normalized > -bound)
        non_omitted_nums = tf.reduce_sum(tf.cast(non_omitted_bool, tf.float32), axis=compress_axes)
        omitted_inputs = tf.where(non_omitted_bool, inputs, tf.zeros_like(inputs, dtype=tf.float32))
        omitted_batch_mean = tf.reduce_sum(omitted_inputs, axis=compress_axes) / non_omitted_nums
        omitted_batch_variance = (tf.reduce_sum(tf.square(omitted_inputs), axis=compress_axes) - tf.square(omitted_batch_mean)) / non_omitted_nums

        update_mean_op = tf.assign(mean, decay * mean + (1 - decay) * omitted_batch_mean)
        update_variance_op = tf.assign(variance, decay * variance + (1 - decay) * omitted_batch_variance)
        with tf.control_dependencies([update_mean_op, update_variance_op]):
            normed = tf.nn.batch_normalization(inputs, omitted_batch_mean, omitted_batch_variance, beta, gamma, variance_epsilon=eps)
            return normed

    def test():
        normed = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon=eps)
        return normed

    rigid_normed = tf.cond(training_ph, train, test)

    reg_recognized = tf.nn.relu(rigid_normed - bound) + tf.nn.relu(- rigid_normed - bound)
    reg_sum = tf.reduce_sum(tf.square(reg_recognized))

    return tf.clip_by_value(rigid_normed, -bound, bound), reg_sum
