# good
# finger
# RUN ON RL 3
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1030 --weight_decay 0.001 --seed 20 --max_timesteps 500000
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1030 --weight_decay 0.001 --seed 30 --max_timesteps 500000


# walker
# CTHULHU 4
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --weight_decay 0.001 --seed 10 --max_timesteps 1000000
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --weight_decay 0.001 --seed 20 --max_timesteps 1000000

# walker
# cthulhu 1
CUDA_VISIBLE_DEVICES=5 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --weight_decay 0.00001 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1038 --weight_decay 0.00001 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 10 --max_timesteps 2000000 &
CUDA_VISIBLE_DEVICES=5 taskset -c 9-11 python main.py --policy PytorchSAC --env dm.walker.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 20 --max_timesteps 2000000 &


###########################################################
###########################################################
###########################################################

# still going
# quadruped run (definitely needs 2M)
# A
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.001 --seed 10 --max_timesteps 2000000 &
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.001 --seed 20 --max_timesteps 2000000 &

# B
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.001 --seed 30 --max_timesteps 2000000 &
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.00001 --seed 10 --max_timesteps 2000000 &

# C
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.00001 --seed 20 --max_timesteps 2000000 &
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.00001 --seed 30 --max_timesteps 2000000 &

# D
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 10 --max_timesteps 2000000 &
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 20 --max_timesteps 2000000 &

# quadruped walk (2M)
# E
python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.001 --seed 30 --max_timesteps 2000000 &
python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.001 --seed 40 --max_timesteps 2000000 &

# F
python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.00001 --seed 10 --max_timesteps 2000000 &
python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.00001 --seed 20 --max_timesteps 2000000 &

# G
python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1065 --weight_decay 0.00001 --seed 30 --max_timesteps 2000000 &
python main.py --policy PytorchSAC --env dm.quadruped.walk --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 30 --max_timesteps 2000000 &

# humanoid ablations (5M)
# H -- problem
python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024 --mlp_qf --seed 10 --max_timesteps 5000000 &
python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024 --mlp_policy --seed 10 --max_timesteps 5000000 &

# I -- problem
python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024 --mlp_qf --seed 20 --max_timesteps 5000000 &
python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --train_B --sigma 0.01 --fourier_dim 1024 --mlp_policy --seed 20 --max_timesteps 5000000 &

# J -- problem
python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --train_B --sigma 0.01 --fourier_dim 1024 --seed 10 --max_timesteps 5000000 &
python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --sigma 0.01 --fourier_dim 1024 --seed 10 --max_timesteps 5000000 &

# K -- problem
python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --train_B --sigma 0.01 --fourier_dim 1024 --seed 20 --max_timesteps 5000000 &
python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class FourierMLP --concatenate_fourier --sigma 0.01 --fourier_dim 1024 --seed 20 --max_timesteps 5000000 &


# acrobot
# nian
CUDA_VISIBLE_DEVICES=7 taskset -c 0-3 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1027 --weight_decay 0.001 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c 4-7 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1027 --weight_decay 0.001 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 8-11 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1027 --weight_decay 0.00001 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=7 taskset -c 12-15 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1027 --weight_decay 0.00001 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=8 taskset -c 16-19 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=9 taskset -c 20-23 python main.py --policy PytorchSAC --env dm.acrobot.swingup --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 20 --max_timesteps 1000000 &

# hopper
# rl 3
CUDA_VISIBLE_DEVICES=0 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --weight_decay 0.001 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --weight_decay 0.001 --seed 20 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=0 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --weight_decay 0.00001 --seed 10 --max_timesteps 1000000 &
CUDA_VISIBLE_DEVICES=1 python main.py --policy PytorchSAC --env dm.hopper.hop --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1033 --weight_decay 0.00001 --seed 20 --max_timesteps 1000000 &



# humanoid (5M)
# PARIS
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1064 --weight_decay 0.001 --seed 10 --max_timesteps 5000000
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --first_dim 1064 --weight_decay 0.001 --seed 20 --max_timesteps 5000000
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 10 --max_timesteps 5000000
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env dm.humanoid.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-4 --network_class LogUniformFourierMLP --fourier_dim 1024 --sigma 0.01 --seed 20 --max_timesteps 5000000
