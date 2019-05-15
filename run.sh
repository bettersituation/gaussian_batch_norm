#!/usr/bin/env bash

# run like this
python run.py --data_type mnist --net_name simple --batch_norm none --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 50
python run.py --data_type mnist --net_name simple --batch_norm batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 50
python run.py --data_type mnist --net_name simple --batch_norm rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 50
python run.py --data_type mnist --net_name simple --batch_norm clipped_rigid_batch_norm --bound 5 --reg_cf 1 --lr 0.001 --batch_size 100 --epoch 50
