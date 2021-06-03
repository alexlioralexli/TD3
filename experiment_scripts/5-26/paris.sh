CUDA_VISIBLE_DEVICES=1 taskset -c 12-14 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1064 --weight_decay 0.00001 --seed 10 --max_timesteps 5000000 --lr 1e-4 &
CUDA_VISIBLE_DEVICES=1 taskset -c 15-17 python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1064 --weight_decay 0.00001 --seed 20 --max_timesteps 5000000 --lr 1e-4 &

