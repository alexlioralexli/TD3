envs = [
    'dm.quadruped.run',
    'dm.humanoid.run',
]

total = 0
lr = '1e-4'
for env in envs:
    commands = [f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr {lr} --max_timesteps 1e7"]
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr {lr} --max_timesteps 1e7"
    fourier_dim = 1024
    sigma = 0.01
    commands.append(
        base_str + f' --network_class FourierMLP --concatenate_fourier --train_B --sigma {sigma} --fourier_dim {fourier_dim}')
    count = 0
    for command in commands:
        gpus = list(range(10))
        for seed in [10, 20, 30]:
            if total % 3 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} taskset -c a-b {command} --seed {seed} &')
            count = (count + 1) % len(gpus)

