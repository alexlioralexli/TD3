# cheetah
CUDA_VISIBLE_DEVICES=0 taskset -c 0-2 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 3-5 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=2 taskset -c 6-8 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 9-23 python main.py --policy PytorchSAC --env dm.cheetah.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.001 --fourier_dim 64 --concatenate_fourier --train_B --seed 10 &

# finger
CUDA_VISIBLE_DEVICES=0 taskset -c 12-12 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c 14-16 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c 17-19 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c 20-22 python main.py --policy PytorchSAC --env dm.finger.turn_hard --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &

# fish swim
CUDA_VISIBLE_DEVICES=0 taskset -c 23-25 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
CUDA_VISIBLE_DEVICES=1 taskset -c 26-28 python main.py --policy PytorchSAC --env dm.fish.swim --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &

# quadruped run
CUDA_VISIBLE_DEVICES=2 taskset -c 29-31 python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --seed 20 &
