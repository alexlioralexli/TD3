#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --max_timesteps 1e7 --seed 10 &
CUDA_VISIBLE_DEVICES=0 taskset -c 8-15 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --max_timesteps 1e7 --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 16-23 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --max_timesteps 1e7 --seed 30 &
