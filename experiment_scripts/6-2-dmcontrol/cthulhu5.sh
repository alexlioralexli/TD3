#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 taskset -c 30-32 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=2 taskset -c 33-35 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=3 taskset -c 36-38 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.003 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=4 taskset -c 39-41 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.003 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=1 taskset -c 42-44 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=2 taskset -c 45-47 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=3 taskset -c 48-50 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.03 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=4 taskset -c 51-53 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.03 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &