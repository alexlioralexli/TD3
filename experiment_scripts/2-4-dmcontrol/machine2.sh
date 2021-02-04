#!/usr/bin/env bash
# quadruped run
CUDA_VISIBLE_DEVICES=0 taskset -c 0-3 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=1 taskset -c 4-7 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &

# quadruped walk
CUDA_VISIBLE_DEVICES=2 taskset -c 8-11 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 12-15 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 16-19 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &

# hopper hop
CUDA_VISIBLE_DEVICES=1 taskset -c 20-23 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 24-27 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &

CUDA_VISIBLE_DEVICES=3 taskset -c 28-31 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 32-35 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 30 &

CUDA_VISIBLE_DEVICES=1 taskset -c 36-39 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 40-44 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 45-49 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 128 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=0 taskset -c 50-54 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 128 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 55-59 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 60-63 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &
