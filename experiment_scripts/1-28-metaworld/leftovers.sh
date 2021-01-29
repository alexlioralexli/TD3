CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-open-v1 --n_hidden 3 --hidden_dim 400 --first_dim 1024 --batch_size 128 --start_timesteps 5000 --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-open-v1 --n_hidden 3 --hidden_dim 400 --first_dim 1024 --batch_size 128 --start_timesteps 5000 --seed 20 &
CUDA_VISIBLE_DEVICES=2 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-open-v1 --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-open-v1 --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=4 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-open-v1 --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-open-v1 --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=0 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-close-v1 --n_hidden 3 --hidden_dim 400 --first_dim 1024 --batch_size 128 --start_timesteps 5000 --seed 10 &
CUDA_VISIBLE_DEVICES=1 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-close-v1 --n_hidden 3 --hidden_dim 400 --first_dim 1024 --batch_size 128 --start_timesteps 5000 --seed 20 &
56
CUDA_VISIBLE_DEVICES=2 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-close-v1 --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=3 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-close-v1 --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &
CUDA_VISIBLE_DEVICES=4 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-close-v1 --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 10 &
CUDA_VISIBLE_DEVICES=5 taskset -c a-b python main.py --policy PytorchSAC --env mw.window-close-v1 --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --seed 20 &