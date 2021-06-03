# paris
CUDA_VISIBLE_DEVICES=0 taskset -c 21-23 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --weight_decay 1e-05 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=2 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --weight_decay 1e-05 --lr 1e-4 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=0 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --weight_decay 1e-05 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=2 taskset -c 18-20 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --weight_decay 1e-05 --lr 1e-4 --seed 30 --max_timesteps 1000000 &

# rome
CUDA_VISIBLE_DEVICES=0 taskset -c 0-3 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 86 --weight_decay 1e-05 --lr 1e-4 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=1 taskset -c 4-7 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 86 --weight_decay 1e-05 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=2 taskset -c 8-11 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 86 --weight_decay 1e-05 --lr 1e-4 --seed 30 --max_timesteps 1000000 &


# rl 3
CUDA_VISIBLE_DEVICES=0 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1030 --weight_decay 1e-05 --lr 1e-4 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1030 --weight_decay 1e-05 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=0 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1030 --weight_decay 1e-05 --lr 1e-4 --seed 30 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --weight_decay 1e-05 --lr 1e-4 --seed 10 --max_timesteps 1000000 &


# cthulhu 4
CUDA_VISIBLE_DEVICES=8 taskset -c 45-47 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1027 --weight_decay 1e-05 --lr 1e-4 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 48-50 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1027 --weight_decay 1e-05 --lr 1e-4 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c 51-53 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1027 --weight_decay 1e-05 --lr 1e-4 --seed 30 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 54-56 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --weight_decay 1e-05 --lr 1e-4 --seed 30 --max_timesteps 1000000 &


rsyncparis /home/alexli/workspace/TD3/data/dm.walker.run-PytorchSAC-MLP-05-27
rsyncparis /home/alexli/workspace/TD3/data/dm.hopper.hop-PytorchSAC-MLP-05-27
rsyncparis /home/alexli/workspace/TD3/data/dm.humanoid.run-PytorchSAC-MLP-05-26

rsyncrome /home/alexli/workspace/TD3/data/dm.cheetah.run-PytorchSAC-MLP-05-27

rsyncrl 3 /home/alexli/workspace/TD3/data/dm.finger.turn-hard-PytorchSAC-MLP-05-27
rsyncrl 3 /home/alexli/workspace/TD3/data/dm.walker.run-PytorchSAC-MLP-05-27

rsynccthulhu 4 /home/pathak-visitor1/workspace/TD3/data/dm.acrobot.swingup-PytorchSAC-MLP-05-26
rsynccthulhu 4 /home/pathak-visitor1/workspace/TD3/data/dm.walker.run-PytorchSAC-MLP-05-26
