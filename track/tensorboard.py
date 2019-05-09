import tensorflow as tf


class Tensorboard:
    def __init__(self, board_path):
        self._board_path = board_path
        self.board = tf.summary.FileWriter(board_path, graph=tf.get_default_graph())

    def add_scalar(self, tag, scalar, step, flush=False):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=scalar)])
        self.board.add_summary(summary, step)
        if flush:
            self.board.flush()

    def add_scalars(self, tag_scalar_dict, step, flush=False):
        for tag, scalar in tag_scalar_dict.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=scalar)])
            self.board.add_summary(summary, step)
        if flush:
            self.board.flush()

    def flush(self):
        self.board.flush()
