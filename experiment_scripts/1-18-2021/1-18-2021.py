envs = ['Humanoid-v2', 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v2', 'Walker2d-v2']
for env in envs:
    commands = []
    for depth in ['--n_hidden 1', '--n_hidden 2', '--n_hidden 3']:
        for width in [256, 1024]:
            for omega in [30]:
                commands.append(f'python main.py --env {env} --network_class Siren --omega {omega} {depth} --hidden_dim {width} --expID 18')
    for depth in ['--n_hidden 2', '--n_hidden 4']:
        commands.append(f'python main.py --env {env} --network_class D2RL {depth} --expID 19')
    count = 0
    for command in commands:
        gpus = list(range(0, 5))
        for seed in [10, 20, 30]:
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} {command} --seed {seed} &')
            count = (count + 1) % len(gpus)
