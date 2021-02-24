#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 taskset -c 0-3 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 4-7 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 20 &
CUDA_VISIBLE_DEVICES=3 taskset -c 8-11 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 30 &