#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=8 taskset -c 0-3 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 taskset -c 4-7 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --concatenate_fourier --seed 10 &
CUDA_VISIBLE_DEVICES=8 taskset -c 8-11 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=9 taskset -c 12-15 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024  --seed 10 &
