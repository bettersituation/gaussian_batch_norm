import numpy as np
import tensorflow as tf


class Tensorboard:
    def __init__(self, board_path):
        self._board_path = board_path
        self.board = tf.summary.FileWriter(board_path, graph=tf.get_default_graph())

    def add_scalar(self, tag, scalar, step, flush=False, prefix=None):
        if prefix is not None:
            tag = prefix + '/' + tag
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=scalar)])
        self.board.add_summary(summary, step)
        if flush:
            self.board.flush()

    def add_scalars(self, tag_scalar_dict, step, flush=False, prefix=None):
        for tag, scalar in tag_scalar_dict.items():
            self.add_scalar(tag, scalar, step, flush=False, prefix=prefix)
        if flush:
            self.board.flush()

    def add_histogram(self, tag, values, step, flush=False, bins=100, prefix=None):
        if prefix is not None:
            tag = prefix + '/' + tag
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.board.add_summary(summary, step)
        if flush:
            self.board.flush()

    def add_histograms(self, tag_values_dict, step, flush=False, bins=100, prefix=None):
        for tag, values in tag_values_dict.items():
            self.add_histogram(tag, values, step, flush=False, bins=bins, prefix=prefix)
        if flush:
            self.board.flush()
