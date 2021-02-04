#!/usr/bin/env bash
# humanoid walk
CUDA_VISIBLE_DEVICES=0 taskset -c 0-6 python main.py --policy PytorchSAC --env dm.humanoid.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=1 taskset -c 7-13 python main.py --policy PytorchSAC --env dm.humanoid.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &

# humanoid run
CUDA_VISIBLE_DEVICES=2 taskset -c 14-20 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=3 taskset -c 21-27 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &

# humanoid stand
CUDA_VISIBLE_DEVICES=0 taskset -c 28-33 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 34-39 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 40-45 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 30 &

# swimmer 15
CUDA_VISIBLE_DEVICES=3 taskset -c 49-51 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 52-54 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &
CUDA_VISIBLE_DEVICES=1 taskset -c 55-57 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 58-60 python main.py --policy PytorchSAC --env dm.swimmer.swimmer15 --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 30 &


## ablations
# finger
CUDA_VISIBLE_DEVICES=3 taskset -c 46-48 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --seed 10 &
CUDA_VISIBLE_DEVICES=0 taskset -c 61-63 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --seed 20 &
