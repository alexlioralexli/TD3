# --expID 27
envs = [
'dm.acrobot.swingup',
'dm.cheetah.run',
'dm.hopper.hop',
'dm.quadruped.run',
'dm.walker.run',
'dm.humanoid.run',
'dm.humanoid.stand',
'dm.humanoid.walk',
]

total = 0
for env in envs:
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr 1e-3 --tau 0.01"
    commands = [f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr 1e-3 --tau 0.01"]
    # fourier features
    for i, type in enumerate(['--train_B', '--concatenate_fourier --train_B']):
        for sigma in [0.01, 0.001]:
            if i != 0 or sigma == 0.001:
                commands.append(base_str + f' --network_class FourierMLP --sigma {sigma} --fourier_dim 1024 {type}')
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

print(len(commands))