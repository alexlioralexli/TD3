CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1030 --weight_decay 0.001 --seed 20 --max_timesteps 500000 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1030 --weight_decay 0.001 --seed 30 --max_timesteps 500000 &