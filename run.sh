#!/usr/bin/env bash

# simple comparison mnist
python run.py --data_type mnist --net_name simple --batch_norm none --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 50
python run.py --data_type mnist --net_name simple --batch_norm batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 50
python run.py --data_type mnist --net_name simple --batch_norm rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 50
python run.py --data_type mnist --net_name simple --batch_norm clipped_rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 50

# simple comparison cifar10
python run.py --data_type cifar10 --net_name simple --batch_norm batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 200
python run.py --data_type cifar10 --net_name simple --batch_norm rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 200
python run.py --data_type cifar10 --net_name simple --batch_norm clipped_rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 200

# vgg16 default comparison
python run.py --data_type cifar10 --net_name vgg16 --batch_norm batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 200
python run.py --data_type cifar10 --net_name vgg16 --batch_norm rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 200
python run.py --data_type cifar10 --net_name vgg16 --batch_norm clipped_rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 200

# lr 0.01 comparison
python run.py --data_type cifar10 --net_name vgg16 --batch_norm batch_norm --bound 5 --reg_cf 1 --lr 0.01 --batch_size 100 --epoch 200
python run.py --data_type cifar10 --net_name vgg16 --batch_norm rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.01 --batch_size 100 --epoch 200
python run.py --data_type cifar10 --net_name vgg16 --batch_norm clipped_rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.01 --batch_size 100 --epoch 200

# lr 0.005 comparison
python run.py --data_type cifar10 --net_name vgg16 --batch_norm batch_norm --bound 5 --reg_cf 1 --lr 0.005 --batch_size 100 --epoch 200
python run.py --data_type cifar10 --net_name vgg16 --batch_norm rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.005 --batch_size 100 --epoch 200
python run.py --data_type cifar10 --net_name vgg16 --batch_norm clipped_rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.005 --batch_size 100 --epoch 200
