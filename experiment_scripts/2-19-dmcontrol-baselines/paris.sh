#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-3 --seed 10 &
CUDA_VISIBLE_DEVICES=0 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-3 --seed 20 &
CUDA_VISIBLE_DEVICES=3 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-3 --seed 30 &
CUDA_VISIBLE_DEVICES=0 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-3 --seed 40 &
CUDA_VISIBLE_DEVICES=3 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 3e-3 --seed 50 &