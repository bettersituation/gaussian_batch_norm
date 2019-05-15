from collections import defaultdict
from pathlib import Path
import numpy as np
import tensorflow as tf


def iter_batch(features, labels, batch_size):
    pos = 0
    size = labels.shape[0]
    if size % batch_size != 0:
        raise Exception('batch size must divide size: {}'.format(size))

    while pos != size:
        next_pos = min(pos + batch_size, size)
        iter_features = features[pos:next_pos]
        iter_labels = labels[pos:next_pos]
        pos = next_pos
        yield iter_features, iter_labels


def calc_mean_of_means(means):
    return sum(means) / len(means)


def calc_cross_entropy(predicts, labels):
    loss = labels * np.log(predicts + 1e-8) + (1 - labels) * np.log(1 - predicts + 1e-8)
    return - np.mean(loss.sum(1))


def calc_acc(predicts, labels):
    equals = np.equal(labels.argmax(1), predicts.argmax(1))
    return equals.mean()


def calc_mean(values):
    return np.mean(values)


def calc_max(values):
    return np.max(values)


def calc_min(values):
    return np.min(values)


def calc_norm(values):
    return np.sqrt(np.sum(np.square(values.astype(np.float64))))


def recursive_max(values_dict):
    max_values = dict()
    for k, v in values_dict.items():
        max_values[k] = np.max(v)
    return max_values


def recursive_min(values_dict):
    min_values = dict()
    for k, v in values_dict.items():
        min_values[k] = np.min(v)
    return min_values


def recursive_norm(values_dict):
    norm_values = dict()
    for k, v in values_dict.items():
        norm_values[k] = calc_norm(v)
    return norm_values


def global_max(values_dict):
    max_v = -np.inf
    for v in values_dict.values():
        local_max = calc_max(v)
        if local_max > max_v:
            max_v = local_max
    return max_v


def global_min(values_dict):
    min_v = np.inf
    for v in values_dict.values():
        local_min = calc_min(v)
        if local_min < min_v:
            min_v = local_min
    return min_v


def augment_dict_values(dicts):
    append_dict = defaultdict(list)
    for dt in dicts:
        for k, v in dt.items():
            append_dict[k].append(v)
    aug_dict = {k: np.vstack(v) for k, v in append_dict.items()}
    return aug_dict


def calc_absmax(values):
    return np.max(np.abs(values))


def calc_absmin(values):
    return np.min(np.abs(values))


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
