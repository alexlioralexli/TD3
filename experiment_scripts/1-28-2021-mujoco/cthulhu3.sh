#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 taskset -c 0-2 python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 3 --first_dim 1024 --seed 10 &
CUDA_VISIBLE_DEVICES=6 taskset -c 3-5 python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 3 --first_dim 1024 --seed 20 &
CUDA_VISIBLE_DEVICES=7 taskset -c 6-8 python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=8 taskset -c 9-11 python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=5 taskset -c 12-14 python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=8 taskset -c 15-17 python main.py --policy PytorchSAC --env HalfCheetah-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=6 taskset -c 18-20 python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 3 --first_dim 1024 --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c 21-23 python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 3 --first_dim 1024 --seed 20 &
CUDA_VISIBLE_DEVICES=8 taskset -c 24-26 python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c 27-29 python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=6 taskset -c 30-32 python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c 33-35 python main.py --policy PytorchSAC --env Hopper-v2 --n_hidden 2 --network_class FourierMLP --sigma 3e-05 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=8 taskset -c 36-38 python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 3 --first_dim 1024 --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c 39-42 python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 3 --first_dim 1024 --seed 20 &
CUDA_VISIBLE_DEVICES=6 taskset -c 42-45 python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c 45-48 python main.py --policy PytorchSAC --env Walker2d-v2 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &







