#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 taskset -c 0-2 python main.py --policy PytorchSAC --env Ant-v2 --n_hidden 3 --first_dim 1024 --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 3-5 python main.py --policy PytorchSAC --env Ant-v2 --n_hidden 3 --first_dim 1024 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 6-8 python main.py --policy PytorchSAC --env Ant-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 9-11 python main.py --policy PytorchSAC --env Ant-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 12-14 python main.py --policy PytorchSAC --env Ant-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 15-17 python main.py --policy PytorchSAC --env Ant-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
