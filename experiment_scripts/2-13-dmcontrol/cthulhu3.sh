#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.03 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.03 --fourier_dim 64 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.03 --fourier_dim 128 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=0 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.03 --fourier_dim 128 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=1 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.03 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.03 --fourier_dim 256 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 18-20 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 21-23 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 24-26 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 128 --concatenate_fourier --train_B --seed 10 &