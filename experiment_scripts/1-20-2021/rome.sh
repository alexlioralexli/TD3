#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python main.py --policy SAC --env Humanoid-v2 --network_class D2RL --n_hidden 4 --expID 20 --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy SAC --env Humanoid-v2 --network_class D2RL --n_hidden 4 --expID 20 --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy SAC --env Humanoid-v2 --network_class D2RL --n_hidden 4 --expID 20 --seed 30 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy SAC --env Humanoid-v2 --n_hidden 2 --first_dim 256 --expID 19 --seed 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy SAC --env Humanoid-v2 --n_hidden 2 --first_dim 256 --expID 19 --seed 20 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy SAC --env Humanoid-v2 --n_hidden 2 --first_dim 256 --expID 19 --seed 30 &
CUDA_VISIBLE_DEVICES=2 python main.py --policy SAC --env Humanoid-v2 --n_hidden 2 --first_dim 1024 --expID 19 --seed 10 &
CUDA_VISIBLE_DEVICES=3 python main.py --policy SAC --env Humanoid-v2 --n_hidden 2 --first_dim 1024 --expID 19 --seed 20 &
