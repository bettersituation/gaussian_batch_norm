import tensorflow as tf


def none_batch_norm(scope, inputs, *args):
    return inputs, tf.zeros(shape=tuple(), dtype=tf.float32), inputs


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
        decay = 0.99
        batch_mean, batch_variance = tf.nn.moments(inputs, compress_axes)
        update_mean_op = tf.assign(mean, decay * mean + (1 - decay) * batch_mean)
        update_variance_op = tf.assign(variance, decay * variance + (1 - decay) * batch_variance)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)

        normed = tf.nn.batch_normalization(inputs, mean, variance, 0., 1., variance_epsilon=eps)
        return normed

    def test():
        normed = tf.nn.batch_normalization(inputs, mean, variance, 0., 1., variance_epsilon=eps)
        return normed

    normed = tf.cond(training_ph, train, test)
    bn_normed = gamma * normed + beta
    return bn_normed, tf.zeros(shape=tuple(), dtype=tf.float32), normed


def rigid_batch_norm(scope, inputs, training_ph, bound, *args):
    eps = 1e-8
    ndims = inputs.get_shape().ndims
    compress_axes = [i for i in range(ndims - 1)]
    reduced_axes = [i for i in range(1, ndims)]
    shape = inputs.get_shape().as_list()[-1]
    with tf.variable_scope('{}/bn'.format(scope)):
        gamma = tf.get_variable('gamma', shape, tf.float32, tf.ones_initializer(), trainable=True)
        beta = tf.get_variable('beta', shape, tf.float32, tf.zeros_initializer(), trainable=True)
        mean = tf.get_variable('mean', shape, tf.float32, tf.zeros_initializer(), trainable=False)
        variance = tf.get_variable('variance', shape, tf.float32, tf.ones_initializer(), trainable=False)

    def train():
        decay = 0.99
        batch_mean, batch_variance = tf.nn.moments(inputs, compress_axes)
        pre_normalized = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, 0., 1., variance_epsilon=eps)

        non_omitted_bool = tf.logical_and(pre_normalized < bound, pre_normalized > -bound)
        non_omitted_nums = tf.reduce_sum(tf.cast(non_omitted_bool, tf.float32), axis=compress_axes)
        omitted_inputs = tf.where(non_omitted_bool, inputs, tf.zeros_like(inputs, dtype=tf.float32))
        omitted_batch_mean = tf.reduce_sum(omitted_inputs, axis=compress_axes) / non_omitted_nums

        update_mean_op = tf.assign(mean, decay * mean + (1 - decay) * omitted_batch_mean)
        update_variance_op = tf.assign(variance, decay * variance + (1 - decay) * batch_variance)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)

        normed = tf.nn.batch_normalization(inputs, omitted_batch_mean, batch_variance, 0., 1., variance_epsilon=eps)
        return normed

    def test():
        normed = tf.nn.batch_normalization(inputs, mean, variance, 0., 1., variance_epsilon=eps)
        return normed

    omitted_normed = tf.cond(training_ph, train, test)
    reg_recognized = tf.nn.relu(omitted_normed - bound) + tf.nn.relu(- omitted_normed - bound)
    reg_sum = tf.reduce_mean(tf.reduce_sum(tf.square(reg_recognized), reduced_axes))

    rigid_normed = gamma * omitted_normed + beta

    return rigid_normed, reg_sum, omitted_normed


def clipped_rigid_batch_norm(scope, inputs, training_ph, bound, *args):
    eps = 1e-8
    ndims = inputs.get_shape().ndims
    compress_axes = [i for i in range(ndims - 1)]
    reduced_axes = [i for i in range(1, ndims)]
    shape = inputs.get_shape().as_list()[-1]
    with tf.variable_scope('{}/bn'.format(scope)):
        gamma = tf.get_variable('gamma', shape, tf.float32, tf.ones_initializer(), trainable=True)
        beta = tf.get_variable('beta', shape, tf.float32, tf.zeros_initializer(), trainable=True)
        mean = tf.get_variable('mean', shape, tf.float32, tf.zeros_initializer(), trainable=False)
        variance = tf.get_variable('variance', shape, tf.float32, tf.ones_initializer(), trainable=False)

    def train():
        decay = 0.99
        batch_mean, batch_variance = tf.nn.moments(inputs, compress_axes)
        pre_normalized = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, 0., 1., variance_epsilon=eps)

        non_omitted_bool = tf.logical_and(pre_normalized < bound, pre_normalized > -bound)
        non_omitted_nums = tf.reduce_sum(tf.cast(non_omitted_bool, tf.float32), axis=compress_axes)
        omitted_inputs = tf.where(non_omitted_bool, inputs, tf.zeros_like(inputs, dtype=tf.float32))
        omitted_batch_mean = tf.reduce_sum(omitted_inputs, axis=compress_axes) / non_omitted_nums

        update_mean_op = tf.assign(mean, decay * mean + (1 - decay) * omitted_batch_mean)
        update_variance_op = tf.assign(variance, decay * variance + (1 - decay) * batch_variance)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)

        normed = tf.nn.batch_normalization(inputs, omitted_batch_mean, batch_variance, 0., 1., variance_epsilon=eps)
        return normed

    def test():
        normed = tf.nn.batch_normalization(inputs, mean, variance, 0., 1., variance_epsilon=eps)
        return normed

    omitted_normed = tf.cond(training_ph, train, test)
    reg_recognized = tf.nn.relu(omitted_normed - bound) + tf.nn.relu(- omitted_normed - bound)
    reg_sum = tf.reduce_mean(tf.reduce_sum(tf.square(reg_recognized), reduced_axes))

    rigid_normed = gamma * tf.clip_by_value(omitted_normed, -bound, bound) + beta

    return rigid_normed, reg_sum, omitted_normed
