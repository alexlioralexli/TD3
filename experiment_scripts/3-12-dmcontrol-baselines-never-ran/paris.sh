#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 taskset -c 0-3 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 4-7 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 8-11 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-4 --seed 30 &
CUDA_VISIBLE_DEVICES=3 taskset -c 12-15 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 16-19 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 20-23 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-4 --seed 30 &