#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=6 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 256 --concatenate_fourier --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 256 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 256  --seed 10 &
CUDA_VISIBLE_DEVICES=6 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 1024 --concatenate_fourier --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c 18-20 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=6 taskset -c 21-23 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 1024  --seed 10 &
CUDA_VISIBLE_DEVICES=7 taskset -c 24-26 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --concatenate_fourier --train_B --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &

