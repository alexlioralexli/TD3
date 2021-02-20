#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=8 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-3 --seed 40 &
CUDA_VISIBLE_DEVICES=9 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-3 --seed 50 &
CUDA_VISIBLE_DEVICES=8 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --seed 10 &
CUDA_VISIBLE_DEVICES=9 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --seed 20 &
CUDA_VISIBLE_DEVICES=8 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --seed 30 &
CUDA_VISIBLE_DEVICES=9 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --seed 40 &
CUDA_VISIBLE_DEVICES=8 taskset -c 18-20 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --seed 50 &