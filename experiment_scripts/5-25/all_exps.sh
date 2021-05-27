




# cheetah
CUDA_VISIBLE_DEVICES=8 taskset -c a-b python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 86 --weight_decay 0.001 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c a-b python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 86 --weight_decay 0.001 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c a-b python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 86 --weight_decay 0.00001 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c a-b python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 86 --weight_decay 0.00001 --seed 20 --max_timesteps 1000000 &
