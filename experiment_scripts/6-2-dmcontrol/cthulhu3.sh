#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=4 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=7 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 30 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=3 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=4 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=7 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 30 --max_timesteps 1000000 &