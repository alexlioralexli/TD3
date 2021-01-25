# --expID 27
envs = [
'dm.acrobot.swingup',
'dm.cheetah.run',
'dm.finger.turn_hard',
'dm.fish.swim',
'dm.hopper.hop',
'dm.quadruped.run',
'dm.quadruped.walk',
'dm.swimmer.swimmer15',
'dm.swimmer.swimmer6',
'dm.walker.run',
'dm.humanoid.run',
'dm.humanoid.stand',
'dm.humanoid.walk',
]
total = 0
for env in envs:
    base_str = f'python main.py --policy PytorchSAC --automatic_entropy_tuning --n_hidden 2 --hidden_dim 1024 --env {env}'
    commands = [base_str]
    # fourier features
    for type in ['--train_B', '--concatenate_fourier --train_B']:
        for sigma in [0.01, 0.001]:
            commands.append(base_str + f' --network_class FourierMLP --sigma {sigma} --fourier_dim 1024 {type}')
    count = 0
    for command in commands:
        # gpus = list(range(8,10))
        gpus = [1,3,4,5]
        for seed in [10, 20]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} {command} --seed {seed} &')
            count = (count + 1) % len(gpus)
