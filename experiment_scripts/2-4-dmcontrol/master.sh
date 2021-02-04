#!/usr/bin/env bash

# cheetah
CUDA_VISIBLE_DEVICES=0 taskset -c 0-3 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 4-7 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 8-11 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 128 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 12-15 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 128 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=0 taskset -c 16-19 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 20-23 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &

# finger
CUDA_VISIBLE_DEVICES=2 taskset -c 24-27 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 28-31 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 32-35 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &
CUDA_VISIBLE_DEVICES=1 taskset -c 36-39 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 40-43 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=3 taskset -c 44-47 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 30 &

# fish swim
CUDA_VISIBLE_DEVICES=0 taskset -c 48-51 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=1 taskset -c 52-55 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &
CUDA_VISIBLE_DEVICES=2 taskset -c 56-59 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=3 taskset -c 60-63 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 30 &

# quadruped run
CUDA_VISIBLE_DEVICES=0 taskset -c 0-3 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=1 taskset -c 4-7 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &

# quadruped walk
CUDA_VISIBLE_DEVICES=2 taskset -c 8-11 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 12-15 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 16-19 python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &

# hopper hop
CUDA_VISIBLE_DEVICES=1 taskset -c 20-23 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 24-27 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 30 &

CUDA_VISIBLE_DEVICES=3 taskset -c 28-31 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 32-35 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 30 &

CUDA_VISIBLE_DEVICES=1 taskset -c 36-39 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 40-44 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 45-49 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 128 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=0 taskset -c 50-54 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 128 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 55-59 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 60-63 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &

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


CUDA_VISIBLE_DEVICES=0 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --seed 30 &
CUDA_VISIBLE_DEVICES=1 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=3 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 30 &

# walker
CUDA_VISIBLE_DEVICES=0 taskset -c 12-15 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 16-18 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 20-23 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --seed 30 &

# quadruped run
CUDA_VISIBLE_DEVICES=3 taskset -c 24-27 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=0 taskset -c 28-31 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=1 taskset -c 32-35 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 30 &
CUDA_VISIBLE_DEVICES=2 taskset -c 36-39 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 40-43 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c 44-47 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --seed 30 &

# humanoid stand
CUDA_VISIBLE_DEVICES=1 taskset -c 48-53 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 54-58 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --seed 20 &
CUDA_VISIBLE_DEVICES=3 taskset -c 59-63 python main.py --policy PytorchSAC --env dm.humanoid.stand --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --seed 30 &

