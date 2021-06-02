#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=3 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.001 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=4 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.003 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=5 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.003 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=6 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=7 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c 18-20 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.03 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 21-23 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.03 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=2 taskset -c 24-26 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.1 --fourier_dim 1024  --seed 40 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=3 taskset -c 27-29 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.1 --fourier_dim 1024  --seed 50 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=4 taskset -c 30-32 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 10 --max_timesteps 2000000 &
CUDA_VISIBLE_DEVICES=5 taskset -c 33-35 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 20 --max_timesteps 2000000 &
CUDA_VISIBLE_DEVICES=6 taskset -c 36-38 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 30 --max_timesteps 2000000 &
CUDA_VISIBLE_DEVICES=7 taskset -c 39-41 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 10 --max_timesteps 2000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c 42-44 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 20 --max_timesteps 2000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 45-47 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024  --seed 30 --max_timesteps 2000000 &
