envs = [
    'dm.acrobot.swingup',
    'dm.cheetah.run',
    'dm.finger.turn_hard',
    'dm.walker.run',
    'dm.quadruped.run',
    'dm.quadruped.walk',
    'dm.hopper.hop',
    'dm.humanoid.run',
]

lff_configs = [
    (1024, 0.01),
    (64, 0.003),    # 64, 0.003
    (1024, 0.01),
    (1024, 0.001),
    (1024, 0.01),
    (1024, 0.01),
    (1024, 0.001),
    (1024, 0.01)
]

times = [
    1e6, 1e6, 1e6, 1e6, 2e6, 2e6, 1e6, 5e6
]

sigmas = [0.001, 0.003, 0.03, 0.1]
# sigmas = [0.001, 0.003, 0.01, 0.03, 0.1]

lr = '1e-4'
count = 0
for i, env in enumerate(envs):
    commands = []
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr {lr}"
    max_timesteps = int(times[i])

    # LFF
    fourier_dim, sigma = lff_configs[i]
    for sigma_sweep in sigmas:
        # if sigma_sweep == sigma:
        for seed in [10, 20, 30]:
            commands.append(base_str + f' --network_class FourierMLP --concatenate_fourier --train_B'
                                       f' --sigma {sigma_sweep} --fourier_dim {fourier_dim}  --seed {seed}')

    for command in commands:
        count += 1
        print(f'CUDA_VISIBLE_DEVICES=0 taskset -c a-b {command} --max_timesteps {max_timesteps} &')
        if count % 10 == 0:
            print(count)
print(count)
# ablation_envs = []
