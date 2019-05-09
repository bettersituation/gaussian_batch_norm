import tensorflow as tf
import numpy as np


class Loader:
    def __init__(self, data_type='cifar10'):
        self._data_type = data_type

        if data_type == 'mnist':
            self._raw_train_data, self._raw_test_data = tf.keras.datasets.mnist.load_data()
        elif data_type == 'cifar10':
            self._raw_train_data, self._raw_test_data = tf.keras.datasets.cifar10.load_data()
        elif data_type == 'cifar100':
            self._raw_train_data, self._raw_test_data = tf.keras.datasets.cifar100.load_data()

        self._train_size = self._raw_train_data[1].shape[0]
        self._test_size = self._raw_test_data[1].shape[0]

        self._feature_shape = self._raw_test_data[0][0].shape
        self._feature_num = self._raw_test_data[0][0].size
        self._label_num = self._raw_test_data[1].max() + 1

        self.train_feature = None
        self.train_label = None
        self.test_feature = None
        self.test_label = None
        self.train_checking = 0
        self.epoch_count = 0
        self._process_feature()
        self._process_label()

    def _process_feature(self):
        train_feature = self._raw_train_data[0]
        train_feature = train_feature / 255. - 0.5
        self.train_feature = train_feature

        test_feature = self._raw_test_data[0]
        test_feature = test_feature / 255. - 0.5
        self.test_feature = test_feature

    def _process_label(self):
        train_label = self._raw_train_data[1]
        one_hot_train_label = np.zeros([self._train_size, self._label_num], np.float32)
        one_hot_train_label[np.arange(self._train_size), train_label] = 1.
        self.train_label = one_hot_train_label

        test_label = self._raw_test_data[1]
        one_hot_test_label = np.zeros([self._test_size, self._label_num], np.float32)
        one_hot_test_label[np.arange(self._test_size), test_label] = 1.
        self.test_label = one_hot_test_label

    def get_shape(self, flatten=False):
        if flatten:
            return [self._feature_num], self._label_num
        else:
            return list(self._feature_shape), self._label_num

    def get_train_batch(self, batch_size, flatten=False):
        if self.train_checking + batch_size <= self._train_size:
            x = self.train_feature[self.train_checking:self.train_checking + batch_size]
            y = self.train_label[self.train_checking:self.train_checking + batch_size]
            self.train_checking += batch_size
        else:
            start_nums = self.train_checking + batch_size - self._train_size
            x = np.vstack([self.train_feature[self.train_checking:], self.train_feature[:start_nums+1]])
            y = np.vstack([self.train_label[self.train_checking:], self.train_label[:start_nums+1]])
            self.train_checking = start_nums
            print('new epoch', self.epoch_count)

        if flatten:
            x = x.reshape(batch_size, self._feature_num)
        else:
            if x.ndim == 3:
                x = x[..., np.newaxis]
        return x, y

    def get_test_batch(self, flatten=False):
        if flatten:
            return self.test_feature.reshape(self._test_size, self._feature_num), self.test_label
        else:
            if self.test_feature.ndim == 3:
                return self.test_feature[..., np.newaxis], self.test_label
            else:
                return self.test_feature, self.test_label
