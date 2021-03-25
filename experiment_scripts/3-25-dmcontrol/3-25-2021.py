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
    for extra in ['--concatenate_fourier --train_B',
                  '--concatenate_fourier',
                  '--train_B',
                  '']:
        commands.append(base_str + f' --network_class LogUniformFourierMLP --fourier_dim {fourier_dim} {extra}')
    count = 0
    for command in commands:
        gpus = list(range(10))
        for seed in [10]:
            if total % 3 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} taskset -c a-b {command} --seed {seed} &')
            count = (count + 1) % len(gpus)

