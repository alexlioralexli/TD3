# expID should start with 19

envs = ['Humanoid-v2', 'Ant-v3', 'HalfCheetah-v3']
for env in envs:
    commands = []
    for depth in ['--n_hidden 4']:
        commands.append(f'python main.py --policy SAC --env {env} --network_class D2RL {depth} --expID 20')
    # normal mlp
    for depth in ['--n_hidden 2', '--n_hidden 3']:
        for first_dim in [256, 1024]:
            commands.append(f'python main.py --policy SAC --env {env} {depth} --first_dim {first_dim} --expID 19')
    # fourier features
    for depth in ['--n_hidden 1', '--n_hidden 2']:
        for type in ['--train_B --expID 21', '--concatenate_fourier --train_B --expID 22']:
            fourier_dim = 1024
            for sigma in [0.01, 0.006, 0.001]:
                commands.append(f'python main.py --policy SAC --env {env} --network_class FourierMLP --sigma {sigma} --fourier_dim {fourier_dim} {depth} {type}')
    count = 0
    for command in commands:
        gpus = list(range(8,10))
        for seed in [10, 20, 30]:
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} {command} --seed {seed} &')
            count = (count + 1) % len(gpus)
