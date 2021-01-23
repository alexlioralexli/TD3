#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=7 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 python main.py --policy SAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.humanoid.walk --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &