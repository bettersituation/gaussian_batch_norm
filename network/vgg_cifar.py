from functools import partial
from batch_norm.batch_norm_types import *
import tensorflow as tf


class VGG:
    def __init__(sess, labels_num, vgg_name, batch_norm_type, bound, clip_value, reg_cf, lr):
        self.sess = sess
        self.labels_num = labels_num
        self.vgg_name = vgg_name
        self.batch_norm_type = batch_norm_type
        self.bound = bound
        self.clip_value = clip_value
        self.reg_cf = reg_cf
        self.lr = lr
        self.inputs_ph = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='inputs')
        self.labels_ph = tf.placeholder(dtype=tf.float32, shape=[None, labels_num], name='labels')
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name='training')
        self.values_dict = dict()
        self.values_dict['bn_values'] = dict()
        self.values_dict['gradients'] = dict()
        self._set_graph()
        self._set_gradients()

    def train(x, y):
        feed_dict = {self.inputs_ph: x, self.labels_ph: y, self.training_ph: True}
        targets = [self.values_dict['loss'], self.train_op]
        loss, _ = self.sess.run(targets, feed_dict = feed_dict)
        return loss

    def test(x):
        feed_dict = {self.inputs_ph: x, self.training_ph: False}
        loss, predicts = self.sess.run(self.values_dict['predicts'], feed_dict=feed_dict)
        return predicts

    def get_bn_values(x):
        feed_dict = {self.inputs_ph: x, self.training_ph: False}
        bn_values = self.sess.run(self.values_dict['bn_values'], feed_dict=feed_dict)
        return bn_values

    def get_gradients(x, y):
        feed_dict = {self.inputs_ph: x, self.labels_ph: y, self.training_ph: True}
        gradients = self.sess.run(self.values_dict['gradients'], feed_dict = feed_dict)
        return gradients

    def _set_graph():
        if vgg_name == 'vgg16':
            self._set_vgg16()
        elif vgg_name == 'vgg19':
            self._set_vgg19()

        opt = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = opt.minimize(self.values_dict['loss_with_reg'])

    def _set_vgg16():
        self.values_dict['reg_loss'] = 0.
        dense = partial(tf.layers.dense, activation=tf.nn.relu)
        last_dense = partial(tf.layers.dense, activation=tf.nn.softmax)
        conv = partial(tf.layers.conv2d, kernel_size=(3, 3), strides=1, padding='same', activation=tf.nn.relu)
        max_pool = partial(tf.layers.max_pooling2d, pool_size=(2, 2), strides=2)
        bn_func = self._get_bn_func()
        store_bn = self.store_bn_values
        inputs = self.inputs_ph
        for i, layers in enumerate([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']):
            if isinstance(layers, int):
                inputs, reg_loss = bn_func(conv(inputs, layers))
                self.values_dict['reg_loss'] += reg_loss
                store_bn('conv_{}'.format(i), inputs)
            else:
                inputs = max_pool(inputs)

        inputs = inputs.flatten()
        for i, layers in enumerate([512, 512]):
            inputs, reg_loss = bn_func(dense(inputs, layers))
            self.values_dict['reg_loss'] += reg_loss
            store_bn('dense_{}'.format(i), inputs)

        predicts = last_dense(inputs, self.labels_num)
        self.values_dict['predicts'] = predicts

        loss = - tf.reduce_sum(self.labels_ph * tf.log(predicts + 1e-8) + (1 - self.labels_ph) * tf.log(1 - predicts + 1e-8))
        self.values_dict['loss'] = loss

        self.values_dict['loss_with_reg'] = loss + self.reg_cf * self.values_dict['reg_loss']

    def _set_vgg19():
        self.values_dict['reg_loss'] = 0.
        dense = partial(tf.layers.dense, activation=tf.nn.relu)
        last_dense = partial(tf.layers.dense, activation=tf.nn.softmax)
        conv = partial(tf.layers.conv2d, kernel_size=(3, 3), strides=1, padding='same', activation=tf.nn.relu)
        max_pool = partial(tf.layers.max_pooling2d, pool_size=(2, 2), strides=2)
        bn_func = self._get_bn_func()
        store_bn = self.store_bn_values
        inputs = self.inputs_ph
        for layers in enumerate([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                                 512, 512, 512, 512, 'M']):
            if isinstance(layers, int):
                inputs, reg_loss = bn_func(conv(inputs, layers))
                self.values_dict['reg_loss'] += reg_loss
                store_bn('conv_{}'.format(i), inputs)
            else:
                inputs = max_pool(inputs)

        inputs = inputs.flatten()
        for i, layers in enumerate([512, 512]):
            inputs, reg_loss = bn_func(dense(inputs, layers))
            self.values_dict['reg_loss'] += reg_loss
            store_bn('dense_{}'.format(i), inputs)

        predicts = last_dense(inputs, self.labels_num)
        self.values_dict['predicts'] = predicts

        loss = - tf.reduce_sum(self.labels_ph * tf.log(predicts + 1e-8) + (1 - self.labels_ph) * tf.log(1 - predicts + 1e-8))
        self.values_dict['loss'] = loss

        self.values_dict['loss_with_reg'] = loss + self.reg_cf * self.values_dict['reg_loss']

    def _set_gradients():
        vs = tf.trainable_variables():
        gs = tf.gradients(self.values_dict['loss_with_reg'], vs)
        for vs, gs in zip(vs, gs):
            self.values_dict['gradients'][vs.name] = gs

    def _get_bn_func():
        if self.batch_norm_type == 'none':
            return none_batch_norm

        if self.batch_norm_type == 'batch_norm':
            return batch_norm

        if self.batch_norm_type == 'rigid_batch_norm':
            return rigid_batch_norm

        if self.batch_norm_type == 'clipped_rigid_batch_norm':
            return partial(clipped_rigid_batch_norm, clip_value=self.clip_value)

    def store_bn_values(key, bn_values):
        self.values_dict['bn_values'][key] = bn_values
