# fair baseline, 20 runs, cthulhu 5
count = 0
envs = ['Ant-v3', 'HalfCheetah-v3']
fair_first_widths = [869, 882]
for env, first_dim in zip(envs, fair_first_widths):
    for depth in ['--n_hidden 2 --expID 12', '--n_hidden 3 --expID 13']:
        for seed in [10, 20, 30, 40, 50]:
            print(f'CUDA_VISIBLE_DEVICES={count % 5} python main.py --env {env} --first_dim {first_dim} {depth} --seed {seed} &')
            count += 1

# other envs
envs = ['Humanoid-v2', 'Hopper-v2', 'Walker2d-v2', 'Reacher-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'dm.finger.spin', 'dm.ball_in_cup.catch']
fair_first_widths = [428, 1014, 992, 1014, 1042, 1014, 1022, 1026]
for env, first_dim in zip(envs, fair_first_widths):
    commands = []
    for depth in ['--n_hidden 2 --expID 12', '--n_hidden 3 --expID 13']:
        commands.append(f'python main.py --env {env} --first_dim {first_dim} {depth} ')
    for depth in ['--n_hidden 1', '--n_hidden 2']:
        for type in ['--train_B --expID 14', '--concatenate_fourier --train_B --expID 15']:
            fourier_dim = 1024
            for sigma in [0.01, 0.006, 0.001]:
                commands.append(f'python main.py --env {env} --network_class FourierMLP --sigma {sigma} --fourier_dim {fourier_dim} {depth} {type}')
    count = 0
    for command in commands:
        # gpus = [5,6,7,8,9]
        # gpus = [4,5,6,8,9]
        # gpus = [0,1,2,3,4]
        gpus = list(range(10))
        for seed in [10, 20, 30, 40, 50]:
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} {command} --seed {seed} &')
            count = (count + 1) % len(gpus)
