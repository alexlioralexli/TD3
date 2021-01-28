#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 taskset -c 0-b python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 3 --first_dim 1024 --seed 10 &
CUDA_VISIBLE_DEVICES=6 taskset -c 3-b python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 3 --first_dim 1024 --seed 20 &
CUDA_VISIBLE_DEVICES=7 taskset -c 6-b python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=8 taskset -c -b python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=5 taskset -c a-b python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=8 taskset -c a-b python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=6 taskset -c a-b python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 3 --first_dim 1024 --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c a-b python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 3 --first_dim 1024 --seed 20 &
CUDA_VISIBLE_DEVICES=8 taskset -c a-b python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c a-b python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=6 taskset -c a-b python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c a-b python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=8 taskset -c a-b python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 3 --first_dim 1024 --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c a-b python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 3 --first_dim 1024 --seed 20 &
CUDA_VISIBLE_DEVICES=6 taskset -c a-b python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c a-b python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &







