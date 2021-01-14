#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --train_B --expID 14 --seed 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --train_B --expID 14 --seed 20 &
CUDA_VISIBLE_DEVICES=3 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --train_B --expID 14 --seed 30 &
CUDA_VISIBLE_DEVICES=1 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.0001 --fourier_dim 1024 --n_hidden 2 --train_B --expID 14 --seed 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.0001 --fourier_dim 1024 --n_hidden 2 --train_B --expID 14 --seed 20 &
CUDA_VISIBLE_DEVICES=3 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.0001 --fourier_dim 1024 --n_hidden 2 --train_B --expID 14 --seed 30 &
CUDA_VISIBLE_DEVICES=1 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 20 &
CUDA_VISIBLE_DEVICES=3 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 30 &
CUDA_VISIBLE_DEVICES=1 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 20 &
CUDA_VISIBLE_DEVICES=3 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 30 &
CUDA_VISIBLE_DEVICES=1 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.0001 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.0001 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 20 &
CUDA_VISIBLE_DEVICES=3 python main.py --env Humanoid-v2 --network_class FourierMLP --sigma 0.0001 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 15 --seed 30 &