#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 30 &
CUDA_VISIBLE_DEVICES=1 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 30 &
CUDA_VISIBLE_DEVICES=1 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 18-20 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 30 &
CUDA_VISIBLE_DEVICES=1 taskset -c 21-23 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 10 &
