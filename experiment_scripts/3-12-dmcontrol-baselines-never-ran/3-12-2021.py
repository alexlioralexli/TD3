envs = [
    # 'dm.acrobot.swingup',
    # 'dm.cheetah.run',
    # 'dm.finger.turn_hard',
    # 'dm.walker.run',
    # 'dm.quadruped.run',
    # 'dm.quadruped.walk',
    'dm.hopper.hop',
    # 'dm.fish.swim',
    # 'dm.humanoid.run',
    # 'dm.humanoid.stand',
    'dm.humanoid.walk',
    # 'dm.swimmer.swimmer15',
]

total = 0
for env in envs:
    lrs = ['1e-4', '3e-4']
    commands = [f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3 --lr {lr}" for lr in lrs]

    count = 0
    for command in commands:
        gpus = list(range(10))
        for seed in [10, 20, 30]:
            if total % 5 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} taskset -c a-b {command} --seed {seed} &')
            count = (count + 1) % len(gpus)
