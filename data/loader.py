import tensorflow as tf
import numpy as np


class Loader:
    def __init__(self, data_type='cifar10'):
        self._data_type = data_type

        if data_type == 'mnist':
            self._raw_train_data, self._raw_test_data = tf.keras.datasets.mnist.load_data()
        elif data_type == 'fashion_mnist':
            self._raw_train_data, self._raw_test_data = tf.keras.datasets.fashion_mnist.load_data()
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

        print('load data:', data_type)
        print('train cases:', self._train_size)
        print('test cases:', self._test_size)

    def _process_feature(self):
        train_feature = self._raw_train_data[0]
        train_feature = 2 * (train_feature / 255.) - 1.
        self.train_feature = train_feature

        test_feature = self._raw_test_data[0]
        test_feature = 2 * (test_feature / 255.) - 1.
        self.test_feature = test_feature

    def _process_label(self):
        train_label = self._raw_train_data[1].reshape(-1)
        one_hot_train_label = np.zeros([self._train_size, self._label_num], np.float32)
        one_hot_train_label[np.arange(self._train_size), train_label] = 1.
        self.train_label = one_hot_train_label

        test_label = self._raw_test_data[1].reshape(-1)
        one_hot_test_label = np.zeros([self._test_size, self._label_num], np.float32)
        one_hot_test_label[np.arange(self._test_size), test_label] = 1.
        self.test_label = one_hot_test_label

    def get_shape(self, flatten=False):
        if flatten:
            return [self._feature_num], self._label_num
        else:
            return list(self._feature_shape), self._label_num

    def get_train_data(self, flatten=False, shuffle=True):
        if shuffle:
            shuffled_indices = np.random.permutation(self._train_size)
            train_features = self.train_feature[shuffled_indices]
            train_labels = self.train_label[shuffled_indices]
        else:
            train_features = self.train_feature
            train_labels = self.train_label

        if flatten:
            return train_features.reshape(self._train_size, self._feature_num), train_labels
        else:
            if train_features.ndim == 3:
                return train_features[..., np.newaxis], train_labels
            else:
                return train_features, train_labels

    def get_test_data(self, flatten=False, shuffle=False):
        if shuffle:
            shuffled_indices = np.random.permutation(self._test_size)
            test_features = self.test_feature[shuffled_indices]
            test_labels = self.test_label[shuffled_indices]
        else:
            test_features = self.test_feature
            test_labels = self.test_label

        if flatten:
            return test_features.reshape(self._test_size, self._feature_num), test_labels
        else:
            if test_features.ndim == 3:
                return test_features[..., np.newaxis], test_labels
            else:
                return test_features, test_labels


if __name__ == '__main__':
    loader = Loader('fashion_mnist')
