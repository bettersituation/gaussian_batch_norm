from functools import partial
import tensorflow as tf


def none_batch_norm(inputs, *args):
    return inputs, 0.


def batch_norm(inputs, training_ph, *args):
    with tf.variable_scope('bn'):
        gamma = tf.get_variable('gamma', shape=inputs.get_shape())
    pass


def rigid_batch_norm(inputs, training_ph, bound, *args):
    pass


def clipped_rigid_batch_norm(inputs, training_ph, bound, clip_value):
    pass
