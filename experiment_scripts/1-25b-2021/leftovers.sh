#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --seed 10 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=6 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=0 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &

CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --seed 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --seed 20 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 10 &
CUDA_VISIBLE_DEVICES=4 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=7 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 20 &

