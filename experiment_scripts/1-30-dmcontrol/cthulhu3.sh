#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 taskset -c 24-26 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --tau 0.01 --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 27-29 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --tau 0.01 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 30-32 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 33-35 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=4 taskset -c 36-38 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c 39-41 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=6 taskset -c 42-44 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c 45-47 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 48-50 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --tau 0.01 --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 51-53 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --tau 0.01 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 18-56 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 21-23 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 20 &