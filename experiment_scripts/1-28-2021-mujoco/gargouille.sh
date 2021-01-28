#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=9 taskset -c 20-22 python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 taskset -c 23-25 python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &

