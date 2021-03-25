#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --concatenate_fourier --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 18-20 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024  --seed 10 &
