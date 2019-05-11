from config.config import *
from data.loader import Loader
from network.vgg_cifar import VGG
from track.tensorboard import Tensorboard
from track.logger import CsvWriter
from util.util import *


def run_func(args):
    set_random_seed(RANDOM_SEED)

    folder_path = DEFAULT_PATH + '/' + args.sub_path
    make_dir(folder_path)

    loader = Loader(args.data_type)
    _, label_num = loader.get_shape()
    test_features, test_labels = loader.get_test_batch()

    tensorboard = Tensorboard(folder_path + '/board')
    csv_writer = CsvWriter(folder_path + '/result.csv')

    columns = ['phase', 'step', 'acc', 'loss', 'bn_max', 'bn_min', 'bn_absmax', 'bn_absmin', 'grad_max', 'grad_min', 'grad_absmax', 'grad_absmin']
    csv_writer.writerow(columns)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = VGG(sess, label_num, args.vgg_name, args.batch_norm, args.bound, args.reg_cf, args.lr)

    for i in range(ITERATION_NUMS + 1):
        if i % RECORD_PERIOD == 0:
            # predicts = net.test(test_features)
            # bn_values = net.get_bn_values(test_features)
            # gradients = net.get_gradients(test_features, test_labels)
            #
            # acc = calc_acc(predicts, test_labels)
            # loss = calc_cross_entropy(predicts, test_labels)
            #
            # bn_max = recursive_max(bn_values)
            # bn_min = recursive_min(bn_values)
            # bn_absmax = recursive_absmax(bn_values)
            # bn_absmin = recursive_absmin(bn_values)
            #
            # grad_max = recursive_max(gradients)
            # grad_min = recursive_min(gradients)
            # grad_absmax = recursive_absmax(gradients)
            # grad_absmin = recursive_absmin(gradients)
            #
            # print('{} step - test - loss: {:.6f} - acc: {:.6f}'.format(i, loss, acc))
            #
            # contents = ['test', i, acc, loss, bn_max, bn_min, bn_absmax, bn_absmin, grad_max, grad_min, grad_absmax, grad_absmin]
            # csv_writer.writerow(contents)
            print('skip')

        # dfsadf
        batch_features, batch_labels = loader.get_train_batch(args.batch_size)

        if i % RECORD_PERIOD == 0:
            bn_values = net.get_bn_values(batch_features)
            gradients = net.get_gradients(batch_features, batch_labels)

            bn_max = recursive_max(bn_values)
            bn_min = recursive_min(bn_values)
            bn_absmax = recursive_absmax(bn_values)
            bn_absmin = recursive_absmin(bn_values)

            grad_max = recursive_max(gradients)
            grad_min = recursive_min(gradients)
            grad_absmax = recursive_absmax(gradients)
            grad_absmin = recursive_absmin(gradients)

            loss, acc = net.train(batch_features, batch_labels)
            print('{} step - train - loss: {:.6f} - acc: {:.6f}'.format(i, loss, acc))

            contents = ['train', i, acc, loss, bn_max, bn_min, bn_absmax, bn_absmin, grad_max, grad_min, grad_absmax, grad_absmin]
            print(contents)
            csv_writer.writerow(contents)

        else:
            net.train(batch_features, batch_labels)

    csv_writer.close()


if __name__ == '__main__':
    class Args:
        sub_path = 'test3'
        data_type = 'cifar10'
        vgg_name = 'vgg16'
        batch_norm = 'rigid_batch_norm'
        bound = 10
        reg_cf = 0.001
        lr = 0.0001
        batch_size = 256

    run_func(Args)
