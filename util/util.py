from pathlib import Path
import numpy as np
import tensorflow as tf


def calc_cross_entropy(predicts, labels):
    loss = labels * np.log(predicts + 1e-8) + (1 - labels) * np.log(1 - predicts + 1e-8)
    return np.sum(loss)


def calc_acc(predicts, labels):
    equals = np.equal(labels.argmax(1), predicts.argmax(1))
    return equals.mean()


def calc_max(values):
    return np.max(values)


def calc_min(values):
    return np.min(values)


def calc_absmax(values):
    return np.max(np.abs(values))


def calc_absmin(values):
    return np.min(np.abs(values))


def calc_norm(values):
    return np.sqrt(np.sum(values.dot(values.T)))


def recursive_max(values_dict):
    max_v = -np.inf
    for v in values_dict.values():
        local_max = calc_max(v)
        if local_max > max_v:
            max_v = local_max
    return max_v


def recursive_min(values_dict):
    min_v = np.inf
    for v in values_dict.values():
        local_min = calc_min(v)
        if local_min < min_v:
            min_v = local_min
    return min_v


def recursive_absmax(values_dict):
    absmax_v = 0.
    for v in values_dict.values():
        local_absmax = calc_absmax(v)
        if local_absmax > absmax_v:
            absmax_v = local_absmax
    return absmax_v


def recursive_absmin(values_dict):
    absmin_v = np.inf
    for v in values_dict.values():
        local_absmin = calc_absmin(v)
        if local_absmin < absmin_v:
            absmin_v = local_absmin
    return absmin_v


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
