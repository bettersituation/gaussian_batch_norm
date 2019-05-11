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

    columns = ['phase', 'step', 'acc', 'loss', 'normed_max', 'normed_min', 'normed_absmax', 'normed_absmin', 'grad_max', 'grad_min', 'grad_absmax', 'grad_absmin']
    csv_writer.writerow(columns)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = VGG(sess, label_num, args.vgg_name, args.batch_norm, args.bound, args.reg_cf, args.lr)

    normed_values_of = []
    gradients_of = []
    accs = []
    losses = []
    normed_maxs = []
    normed_mins = []
    normed_absmaxs = []
    normed_absmins = []
    grad_maxs = []
    grad_mins = []
    grad_absmaxs = []
    grad_absmins = []

    for i in range(ITERATION_NUMS + 1):
        if i % RECORD_PERIOD == 0:
            for batch_features, batch_labels in iter_batch(test_features, test_labels, TEST_BATCH_SIZE):
                predicts = net.test(test_features)
                normed_values = net.get_normed_values(test_features)
                gradients = net.get_gradients(test_features, test_labels)

                acc = calc_acc(predicts, test_labels)
                loss = calc_cross_entropy(predicts, test_labels)

                normed_max = recursive_max(normed_values)
                normed_min = recursive_min(normed_values)
                normed_absmax = recursive_absmax(normed_values)
                normed_absmin = recursive_absmin(normed_values)

                grad_max = recursive_max(gradients)
                grad_min = recursive_min(gradients)
                grad_absmax = recursive_absmax(gradients)
                grad_absmin = recursive_absmin(gradients)

                normed_values_of.append(normed_values)
                gradients_of.append(gradients)
                accs.append(acc)
                losses.append(loss)

                normed_maxs.append(normed_max)
                normed_mins.append(normed_min)
                normed_absmaxs.append(normed_absmax)
                normed_absmins.append(normed_absmin)

                grad_maxs.append(grad_max)
                grad_mins.append(grad_min)
                grad_absmaxs.append(grad_absmax)
                grad_absmins.append(grad_absmin)

            aug_normed_values = augment_dict_values(normed_values_of)
            aug_grad_values = augment_dict_values(gradients_of)
            m_acc = calc_mean_of_means(accs)
            m_loss = calc_mean_of_means(losses)

            m_normed_max = calc_max_of_max(normed_maxs)
            m_normed_min = calc_min_of_min(normed_mins)
            m_normed_absmax = calc_max_of_max(normed_absmaxs)
            m_normed_absmin = calc_min_of_min(normed_absmins)

            m_grad_max = calc_max_of_max(grad_maxs)
            m_grad_min = calc_min_of_min(grad_mins)
            m_grad_absmax = calc_max_of_max(grad_absmaxs)
            m_grad_absmin = calc_min_of_min(grad_absmins)

            log_scalars = {'acc': m_acc,
                           'loss': m_loss,
                           'normed_max': m_normed_max,
                           'normed_min': m_normed_min,
                           'normed_absmax': m_normed_absmax,
                           'normed_absmin': m_normed_absmin,
                           'grad_max': m_grad_max,
                           'grad_min': m_grad_min,
                           'grad_absmax': m_grad_absmax,
                           'grad_absmin': m_grad_absmin
                           }

            tensorboard.add_histograms(aug_normed_values, i, prefix='test')
            tensorboard.add_histograms(aug_grad_values, i, prefix='test')
            tensorboard.add_scalars(log_scalars, i, prefix='test')

            print('{} step - test - loss: {:.6f} - acc: {:.6f}'.format(i, m_loss, m_acc))

            contents = ['test', i, m_acc, m_loss, m_normed_max, m_normed_min, m_normed_absmax, m_normed_absmin, m_grad_max, m_grad_min, m_grad_absmax, m_grad_absmin]
            print(contents)
            csv_writer.writerow(contents)

            normed_values_of.clear()
            gradients_of.clear()
            normed_maxs.clear()
            normed_mins.clear()
            normed_absmaxs.clear()
            normed_absmins.clear()
            grad_maxs.clear()
            grad_mins.clear()
            grad_absmaxs.clear()
            grad_absmins.clear()

        batch_features, batch_labels = loader.get_train_batch(args.batch_size)

        if i % RECORD_PERIOD == 0:
            loss, acc = net.train(batch_features, batch_labels)
            normed_values = net.get_normed_values(batch_features)
            gradients = net.get_gradients(batch_features, batch_labels)

            normed_max = recursive_max(normed_values)
            normed_min = recursive_min(normed_values)
            normed_absmax = recursive_absmax(normed_values)
            normed_absmin = recursive_absmin(normed_values)

            grad_max = recursive_max(gradients)
            grad_min = recursive_min(gradients)
            grad_absmax = recursive_absmax(gradients)
            grad_absmin = recursive_absmin(gradients)

            log_scalars = {'acc': acc,
                           'loss': loss,
                           'normed_max': normed_max,
                           'normed_min': normed_min,
                           'normed_absmax': normed_absmax,
                           'normed_absmin': normed_absmin,
                           'grad_max': grad_max,
                           'grad_min': grad_min,
                           'grad_absmax': grad_absmax,
                           'grad_absmin': grad_absmin
                           }

            tensorboard.add_histograms(normed_values, i, prefix='train')
            tensorboard.add_histograms(grad_values, i, prefix='train')
            tensorboard.add_scalars(log_scalars, i, prefix='train')

            print('{} step - train - loss: {:.6f} - acc: {:.6f}'.format(i, loss, acc))

            contents = ['train', i, acc, loss, normed_max, normed_min, normed_absmax, normed_absmin, grad_max, grad_min, grad_absmax, grad_absmin]
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
