from functools import partial
import tensorflow as tf
from batch_norm.batch_norm_types import *


class VGG:
    def __init__(self, sess, labels_num, vgg_name, batch_norm_type, bound, reg_cf, lr):
        self.sess = sess
        self.labels_num = labels_num
        self.vgg_name = vgg_name
        self.batch_norm_type = batch_norm_type
        self.bound = bound
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
        self.sess.run(tf.global_variables_initializer())

    def train(self, x, y):
        feed_dict = {self.inputs_ph: x, self.labels_ph: y, self.training_ph: True}
        targets = [self.values_dict['loss'], self.values_dict['acc'], self.train_op]
        loss, acc, _ = self.sess.run(targets, feed_dict = feed_dict)
        return loss, acc

    def test(self, x):
        feed_dict = {self.inputs_ph: x, self.training_ph: False}
        predicts = self.sess.run(self.values_dict['predicts'], feed_dict=feed_dict)
        return predicts

    def get_bn_values(self, x):
        feed_dict = {self.inputs_ph: x, self.training_ph: False}
        eval_keys = []
        eval_tensors = []
        for key, tensor in self.values_dict['bn_values'].items():
            eval_keys.append(key)
            eval_tensors.append(tensor)

        bn_values = self.sess.run(eval_tensors, feed_dict=feed_dict)
        return {key: bn_value for key, bn_value in zip(eval_keys, bn_values)}

    def get_gradients(self, x, y):
        feed_dict = {self.inputs_ph: x, self.labels_ph: y, self.training_ph: False}
        eval_keys = []
        eval_tensors = []
        for key, tensor in self.values_dict['gradients'].items():
            eval_keys.append(key)
            eval_tensors.append(tensor)

        gradients = self.sess.run(eval_tensors, feed_dict = feed_dict)
        return {key: grad for key, grad in zip(eval_keys, gradients)}

    def _set_graph(self):
        vgg16_conv_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        vgg19_conv_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        if self.vgg_name == 'vgg16':
            self._set_vgg(vgg16_conv_layers)
        elif self.vgg_name == 'vgg19':
            self._set_vgg(vgg19_conv_layers)

        opt = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = opt.minimize(self.values_dict['loss_with_reg'])

    def _set_vgg(self, conv_layers):
        self.values_dict['reg_loss'] = 0.
        bn_func, use_bias = self._get_bn_func()
        store_bn = self.store_bn_values

        conv = partial(tf.layers.conv2d, kernel_size=(3, 3), strides=1, padding='same', activation=tf.nn.relu, use_bias=use_bias)
        max_pool = partial(tf.layers.max_pooling2d, pool_size=(2, 2), strides=2)

        dense = partial(tf.layers.dense, activation=tf.nn.relu)
        last_dense = partial(tf.layers.dense, activation=tf.nn.softmax)

        inputs = self.inputs_ph
        for i, layers in enumerate(conv_layers):
            if isinstance(layers, int):
                inputs, reg_loss = bn_func('conv_{}'.format(i), conv(inputs, layers), self.training_ph, self.bound)
                self.values_dict['reg_loss'] += reg_loss
                store_bn('conv_{}'.format(i), inputs)
            else:
                inputs = max_pool(inputs)

        inputs = tf.layers.flatten(inputs)
        for i, layers in enumerate([512, 512]):
            inputs, reg_loss = bn_func('dense_{}'.format(i), dense(inputs, layers), self.training_ph, self.bound)
            self.values_dict['reg_loss'] += reg_loss
            store_bn('dense_{}'.format(i), inputs)

        predicts = last_dense(inputs, self.labels_num)
        self.values_dict['predicts'] = predicts

        loss = - tf.reduce_sum(self.labels_ph * tf.log(predicts + 1e-8) + (1 - self.labels_ph) * tf.log(1 - predicts + 1e-8))

        self.values_dict['loss'] = loss
        self.values_dict['loss_with_reg'] = loss + self.reg_cf * self.values_dict['reg_loss']
        self.values_dict['acc'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predicts, 1), tf.argmax(self.labels_ph, 1)), tf.float32))

    def _set_gradients(self):
        vs = tf.trainable_variables()
        gs = tf.gradients(self.values_dict['loss_with_reg'], vs)
        for vs, gs in zip(vs, gs):
            key = vs.name.replace('/', '_').replace(':', '_')
            self.values_dict['gradients'][key] = gs

    def _get_bn_func(self):
        if self.batch_norm_type == 'none':
            return none_batch_norm, True

        if self.batch_norm_type == 'batch_norm':
            return batch_norm, False

        if self.batch_norm_type == 'rigid_batch_norm':
            return rigid_batch_norm, False

    def store_bn_values(self, key, bn_values):
        self.values_dict['bn_values'][key] = bn_values


if __name__ == '__main__':
    vgg19 = VGG(tf.Session(), 100, 'vgg16', 'rigid_batch_norm', 2, 0.2, 0.1)
    for v in tf.trainable_variables():
        print(v)
