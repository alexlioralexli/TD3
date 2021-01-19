#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py --env Humanoid-v2 --first_dim 987 --n_hidden 2 --expID 17  --seed 10 &
CUDA_VISIBLE_DEVICES=2 python main.py --env Humanoid-v2 --first_dim 987 --n_hidden 2 --expID 17  --seed 20 &
CUDA_VISIBLE_DEVICES=1 python main.py --env Humanoid-v2 --first_dim 987 --n_hidden 2 --expID 17  --seed 30 &
CUDA_VISIBLE_DEVICES=2 python main.py --env Humanoid-v2 --first_dim 987 --n_hidden 3 --expID 17  --seed 10 &
CUDA_VISIBLE_DEVICES=1 python main.py --env Humanoid-v2 --first_dim 987 --n_hidden 3 --expID 17  --seed 20 &
CUDA_VISIBLE_DEVICES=2 python main.py --env Humanoid-v2 --first_dim 987 --n_hidden 3 --expID 17  --seed 30 &
