#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --seed 20 &
wait $!
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.humanoid.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.humanoid.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy PytorchSAC --env dm.humanoid.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy PytorchSAC --env dm.humanoid.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &