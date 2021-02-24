#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 18-20 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 20 &
CUDA_VISIBLE_DEVICES=1 taskset -c 21-23 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 30 &