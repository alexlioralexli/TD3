# --expID 27
envs = [
    'HalfCheetah-v2',
    'Hopper-v2',
    'Walker2d-v2',
    'Ant-v2',
    'Humanoid-v2'
]
total = 0
for env in envs:
    base_str = f"python main.py --policy PytorchSAC --env {env} --n_hidden 2"
    commands = [f"python main.py --policy PytorchSAC --env {env} --n_hidden 3 --first_dim 1024"]
    # fourier features
    for type in ['--concatenate_fourier --train_B']:
        for sigma in [0.001, 0.00003]:
            commands.append(base_str + f' --network_class FourierMLP --sigma {sigma} --fourier_dim 1024 {type}')
    count = 0
    for command in commands:
        gpus = list(range(10))
        for seed in [10, 20]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} taskset -c a-b {command} --seed {seed} &')
            count = (count + 1) % len(gpus)
