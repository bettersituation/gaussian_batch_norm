from config.config import *
from data.loader import Loader
from network.net import Net
from track.tensorboard import Tensorboard
from track.values_log import ValuesLog
from track.logger import CsvWriter
from util.util import *


def run_func(args):
    set_random_seed(RANDOM_SEED)

    folder_path = DEFAULT_PATH + '/' + args.sub_path
    make_dir(folder_path + '/board')

    loader = Loader(args.data_type)
    input_shape, label_num = loader.get_shape()
    test_features, test_labels = loader.get_test_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = Net(sess, input_shape, label_num, args.net_name, args.batch_norm, args.bound, args.reg_cf, args.lr)

    normed_keys = net.get_normed_keys()
    gradient_keys = net.get_gradient_keys()

    train_board = Tensorboard(folder_path + '/board/train')
    valid_board = Tensorboard(folder_path + '/board/valid')
    values_log = ValuesLog(normed_keys, gradient_keys)

    csv_writer = CsvWriter(folder_path + '/result.csv')
    columns = ['phase', 'epoch', 'reg_loss', 'loss', 'acc', 'normed_max', 'normed_min', 'gradient_max', 'gradient_min']
    csv_writer.writerow(columns)

    for i in range(1, args.epoch + 1):
        # train
        train_features, train_labels = loader.get_train_data()
        for batch_features, batch_labels in iter_batch(train_features, train_labels, args.batch_size):
            reg_loss, loss, acc, match, normed_values, gradients = net.train(batch_features, batch_labels)

            normed_max_values = recursive_max(normed_values)
            normed_min_values = recursive_min(normed_values)

            grad_max_values = recursive_max(gradients)
            grad_min_values = recursive_min(gradients)
            grad_norm_values = recursive_norm(gradients)

            values_log.save_batchs(reg_loss, loss, acc, normed_max_values, normed_min_values, grad_max_values, grad_min_values, grad_norm_values)

        values_log.calc_batchs()
        reg_loss, loss, acc, normed_max, normed_min, gradient_max, gradient_min, gradient_norm = values_log.get_epochs()
        normed_max_of_max, normed_min_of_min, grad_max_of_max, grad_min_of_min = values_log.get_global_epochs()
        values_log.clear_batch_and_epochs()

        log_scalars = {'reg_loss': reg_loss,
                       'loss': loss,
                       'acc': acc,
                       'normed_max': normed_max_of_max,
                       'normed_min': normed_min_of_min,
                       'gradient_max': grad_max_of_max,
                       'gradient_min': grad_min_of_min,
                       }

        train_board.add_scalars(log_scalars, i, prefix='stats')
        train_board.add_scalars(normed_max, i, prefix='normed_max')
        train_board.add_scalars(normed_min, i, prefix='normed_min')
        train_board.add_scalars(gradient_max, i, prefix='gradient_max')
        train_board.add_scalars(gradient_min, i, prefix='gradient_min')
        train_board.add_scalars(gradient_norm, i, prefix='gradient_norm')

        print('{} epoch - train - reg loss: {:.6f} loss: {:.6f} - acc: {:.6f}'.format(i, reg_loss, loss, acc))

        contents = ['train', i, reg_loss, loss, acc, normed_max_of_max, normed_min_of_min, grad_max_of_max, grad_min_of_min]
        print(contents)
        csv_writer.writerow(contents)

        # valid
        for batch_features, batch_labels in iter_batch(test_features, test_labels, TEST_BATCH_SIZE):
            reg_loss, loss, acc, match, normed_values, gradients = net.test(batch_features, batch_labels)

            normed_max_values = recursive_max(normed_values)
            normed_min_values = recursive_min(normed_values)

            grad_max_values = recursive_max(gradients)
            grad_min_values = recursive_min(gradients)
            grad_norm_values = recursive_norm(gradients)

            values_log.save_batchs(reg_loss, loss, acc, normed_max_values, normed_min_values, grad_max_values, grad_min_values, grad_norm_values)

        values_log.calc_batchs()
        reg_loss, loss, acc, normed_max, normed_min, gradient_max, gradient_min, gradient_norm = values_log.get_epochs()
        normed_max_of_max, normed_min_of_min, grad_max_of_max, grad_min_of_min = values_log.get_global_epochs()
        values_log.clear_batch_and_epochs()

        log_scalars = {'reg_loss': reg_loss,
                       'loss': loss,
                       'acc': acc,
                       'normed_max': normed_max_of_max,
                       'normed_min': normed_min_of_min,
                       'gradient_max': grad_max_of_max,
                       'gradient_min': grad_min_of_min,
                       }

        valid_board.add_scalars(log_scalars, i, prefix='stats')
        valid_board.add_scalars(normed_max, i, prefix='normed_max')
        valid_board.add_scalars(normed_min, i, prefix='normed_min')
        valid_board.add_scalars(gradient_max, i, prefix='gradient_max')
        valid_board.add_scalars(gradient_min, i, prefix='gradient_min')
        valid_board.add_scalars(gradient_norm, i, prefix='gradient_norm')

        print('{} epoch - valid - reg_loss: {:.6f} loss: {:.6f} - acc: {:.6f}'.format(i, reg_loss, loss, acc))

        contents = ['valid', i, reg_loss, loss, acc, normed_max_of_max, normed_min_of_min, grad_max_of_max, grad_min_of_min]
        print(contents)
        csv_writer.writerow(contents)

    csv_writer.close()


if __name__ == '__main__':
    class Args:
        sub_path = 'test0'
        data_type = 'cifar10'
        net_name = 'vgg16'
        batch_norm = 'batch_norm'
        bound = 5
        reg_cf = 1
        lr = 0.01
        batch_size = 200
        epoch = 50

    run_func(Args)
