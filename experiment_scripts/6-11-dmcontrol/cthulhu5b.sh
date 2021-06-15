#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 taskset -c 0-2 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --sigma 0.003 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=5 taskset -c 3-5 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --sigma 0.003 --lr 1e-4 --seed 30 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=6 taskset -c 6-8 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --sigma 0.01 --lr 1e-4 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=7 taskset -c 9-11 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --sigma 0.01 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c 12-14 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --sigma 0.01 --lr 1e-4 --seed 30 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 15-17 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --sigma 0.0001 --lr 1e-4 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=4 taskset -c 18-20 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --sigma 0.0001 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=5 taskset -c 21-23 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --sigma 0.0001 --lr 1e-4 --seed 30 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=6 taskset -c 24-26 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --sigma 0.0003 --lr 1e-4 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=7 taskset -c 27-29 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --sigma 0.0003 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c 30-32 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --sigma 0.0003 --lr 1e-4 --seed 30 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 33-35 python main.py --policy PytorchSAC --network_class VariableInitMLP --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --sigma 0.001 --lr 1e-4 --seed 10 --max_timesteps 1000000 &