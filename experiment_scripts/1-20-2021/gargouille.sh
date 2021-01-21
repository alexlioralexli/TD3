#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class D2RL --n_hidden 4 --expID 20 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class D2RL --n_hidden 4 --expID 20 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class D2RL --n_hidden 4 --expID 20 --seed 30 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 2 --first_dim 256 --expID 19 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 2 --first_dim 256 --expID 19 --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 2 --first_dim 256 --expID 19 --seed 30 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 2 --first_dim 1024 --expID 19 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 2 --first_dim 1024 --expID 19 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 2 --first_dim 1024 --expID 19 --seed 30 &
wait $!
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 3 --first_dim 256 --expID 19 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 3 --first_dim 256 --expID 19 --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 3 --first_dim 256 --expID 19 --seed 30 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 3 --first_dim 1024 --expID 19 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 3 --first_dim 1024 --expID 19 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --n_hidden 3 --first_dim 1024 --expID 19 --seed 30 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 30 &
wait $!
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 30 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 1 --train_B --expID 21 --seed 30 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 30 &
wait $!
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 30 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 1 --concatenate_fourier --train_B --expID 22 --seed 30 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 20 &
wait $!
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 30 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 30 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 2 --train_B --expID 21 --seed 30 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 10 &
wait $!
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 30 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 10 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 20 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.006 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 30 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 10 &
CUDA_VISIBLE_DEVICES=9 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 20 &
CUDA_VISIBLE_DEVICES=8 python main.py --policy SAC --env HalfCheetah-v3 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --n_hidden 2 --concatenate_fourier --train_B --expID 22 --seed 30 &