envs = [
'dm.cheetah.run',
'dm.hopper.hop',
]

total = 0
for env in envs:
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2"
    commands = [
                f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3"]
    # fourier features
    for type in ['--concatenate_fourier --train_B']:
        for sigma in [0.03, 0.01, 0.003]:
            for fourier_dim in [64, 128, 256]:
                commands.append(base_str + f' --network_class FourierMLP --sigma {sigma} --fourier_dim {fourier_dim} {type}')
    count = 0
    for command in commands:
        # gpus = list(range(8,10))
        gpus = list(range(10))
        for seed in [10, 20]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} taskset -c a-b {command} --seed {seed} &')
            count = (count + 1) % len(gpus)
