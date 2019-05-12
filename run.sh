#!/usr/bin/env bash

python run.py --data_type mnist --net_name simple --batch_norm none --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100
python run.py --data_type mnist --net_name simple --batch_norm batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100
python run.py --data_type mnist --net_name simple --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100

python run.py --data_type cifar10 --net_name simple --batch_norm batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100
python run.py --data_type cifar10 --net_name simple --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100

python run.py --data_type cifar10 --net_name vgg16 --batch_norm batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100
python run.py --data_type cifar10 --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100

python run.py --data_type cifar10 --net_name vgg16 --batch_norm batch_norm --bound 10 --reg_cf 1 --lr 0.01 --batch_size 100
python run.py --data_type cifar10 --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.01 --batch_size 100
python run.py --data_type cifar10 --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.005 --batch_size 100

python run.py --data_type cifar100 --net_name vgg16 --batch_norm batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100
python run.py --data_type cifar100 --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100

python run.py --data_type cifar100 --net_name vgg16 --batch_norm batch_norm --bound 10 --reg_cf 1 --lr 0.01 --batch_size 100
python run.py --data_type cifar100 --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.01 --batch_size 100
python run.py --data_type cifar100 --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.005 --batch_size 100

python run.py --data_type fashion_mnist --net_name vgg16 --batch_norm batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100
python run.py --data_type fashion_mnist --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.001 --batch_size 100

python run.py --data_type fashion_mnist --net_name vgg16 --batch_norm batch_norm --bound 10 --reg_cf 1 --lr 0.01 --batch_size 100
python run.py --data_type fashion_mnist --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.01 --batch_size 100
python run.py --data_type fashion_mnist --net_name vgg16 --batch_norm rigid_batch_norm --bound 10 --reg_cf 1 --lr 0.005 --batch_size 100
