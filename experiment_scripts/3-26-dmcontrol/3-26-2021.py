envs = [
    'dm.finger.turn_hard',
    'dm.walker.run',
    'dm.quadruped.run',
    'dm.humanoid.stand',
]

total = 0
lr = '1e-4'
for env in envs:
    commands = []
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr {lr}"
    fourier_dim = 1024
    for extra in ['--concatenate_fourier']:
        for sigma in [10, 100]:
            commands.append(base_str + f' --network_class LogUniformFourierMLP --fourier_dim {fourier_dim} --sigma {sigma} {extra}')
    count = 0
    for command in commands:
        gpus = list(range(10))
        for seed in [10]:
            if total % 3 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} taskset -c a-b {command} --seed {seed} &')
            count = (count + 1) % len(gpus)

print('------------------')
# more seeds for LFF on walker, hopper, cheetah
envs = [
    'dm.cheetah.run', #64, 0.001, 0.003
    'dm.walker.run',  # 1024 0.001
    'dm.hopper.hop', # 1024, 0.001, 0.003
]

configs = [
    [(64, 0.001), (64, 0.003)],
    [(1024, 0.001)],
    [(1024, 0.001), (1024, 0.003)]
]

total = 0
lr = '1e-4'
for env, config in zip(envs, configs):
    commands = []
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr {lr}"
    for (fourier_dim, sigma) in config:
        commands.append(base_str + f' --network_class FourierMLP --concatenate_fourier --train_B --sigma {sigma} --fourier_dim {fourier_dim}')
    count = 0
    for command in commands:
        gpus = list(range(10))
        for seed in [20, 30]:
            if total % 3 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} taskset -c a-b {command} --seed {seed} &')
            count = (count + 1) % len(gpus)