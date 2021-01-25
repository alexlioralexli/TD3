#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.swimmer.swimmer6 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.swimmer.swimmer6 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.swimmer.swimmer6 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --seed 20 &
CUDA_VISIBLE_DEVICES=4 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=5 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=4 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=5 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env dm.walker.run --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
