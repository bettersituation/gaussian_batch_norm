import argparse
from run_func import run_func


def get_parser():
    _data_type = 'cifar10'
    _vgg_name = 'vgg16'
    _bath_norm = 'rigid_batch_norm'
    _bound = 3
    _reg_cf = 1e-3
    _lr = 1e-3
    _batch_size = 256

    parser = argparse.ArgumentParser(description='Run and logging an experiment or a rigid batch norm')
    parser.add_argument('--data_type',
                        default=_data_type,
                        choices=['cifar10', 'cifar100'],
                        help='data type which will be loaded (default: %(default)s) (choices: %(choices)s)'
                        )
    parser.add_argument('--vgg_name',
                        default=_vgg_name,
                        choices=['vgg16, vgg19'],
                        help='model type will be trained (default: %(default)s) (choices: %(choices)s)'
                        )
    parser.add_argument('--batch_norm',
                        default=_bath_norm,
                        choices=['none', 'batch_norm', 'rigid_batch_norm'],
                        help='batch norm which will be used (default: %(default)s) (choices: %(choices)s)'
                        )
    parser.add_argument('--bound',
                        default=_bound,
                        type=float,
                        help='bound of z-score which will be used if rigid batch norm chosen (default: %(default)s)'
                        )
    parser.add_argument('--reg_cf',
                        default=_reg_cf,
                        type=float,
                        help='rigid regularization coefficient which will be used if rigid batch norm chosen (default: %(default)s)'
                        )
    parser.add_argument('--lr',
                        default=_lr,
                        type=float,
                        help='learning rate (default: %(default)s)'
                        )
    parser.add_argument('--batch_size',
                        default=_batch_size,
                        type=int,
                        help='training batch size (default: %(default)s)'
                        )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    kwargs = (args.data_type, args.vgg_name, args.batch_norm, args.bound, args.reg_cf, args.lr, args.batch_size)
    args.sub_path = '{}_{}_{}_bound_{:4.2e}_reg_cf_{:4.2e}_lr_{:4.2e}_batch_{}'.format(*kwargs)

    print('setting as follows')
    for arg in vars(args):
        print('{} : {}'.format(arg, getattr(args, arg)))

    run_func(args)
