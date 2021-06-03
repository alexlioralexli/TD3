"""
SAC state-space: 2M * 8envs (orig 12) * 5seeds * 4 methods (ours vs mlp vs diag fourier vs l2 regul.)
SAC images: 2M * 4envs * 5seeds * 2methods
SAC ablation Value vs Policy vs Both vs Baseline: 2M * 4envs * 5seeds * 2NewMethods
SAC ablation: 1M * 4envs * 3seeds * 3methods (can leave it as is)
SAC no critic: 1M * 4envs * 3seeds * 2methods ---> change to 5seeds and 5M (lower priority)
"""
import math

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
    (64, 0.003),
    (1024, 0.01),
    (1024, 0.001),
    (1024, 0.01),
    (1024, 0.01),
    (1024, 0.001),
    (1024, 0.01)
]

env_dims = [
    (6, 1),
    (17, 6),
    (12, 2),
    (24, 6),
    (78, 12),
    (78, 12),
    (15, 4),
    (67, 21)
]

# LFF: (obs+action) x fourier_dim / 2 + (fourier_dim + obs+action) x hidden_dim
# MLP: (obs + action+1) x first dim + first_dim x hidden_dim
# = first_dim x (hidden_dim + obs + action + 1)
first_dims = []
for i in range(len(env_dims)):
    input_dim = env_dims[i][0] + env_dims[i][1]
    fourier_dim = lff_configs[i][0]
    lff_params = input_dim * (fourier_dim // 2) + (fourier_dim + input_dim) * 1024
    first_dims.append(math.ceil(lff_params / (1024 + 1 + input_dim)))

log_uniform_ranges = [0.01, 0.1]
weight_decays = [1e-3, 3e-3, 1e-2, 3e-2]

lr = '1e-4'
count = 0
for i, env in enumerate(envs):
    commands = []
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr {lr}"
    if i != 7:
        max_timesteps = '2000000'
    else:
        max_timesteps = '5000000'

    # LFF
    # fourier_dim, sigma = lff_configs[i]
    # for seed in [10, 20, 30, 40, 50]:
    #     commands.append(base_str + f' --network_class FourierMLP --concatenate_fourier --train_B'
    #                                f' --sigma {sigma} --fourier_dim {fourier_dim}  --seed {seed}')

    # MLP
    # for seed in [10, 20, 30, 40, 50]:
    #     commands.append(f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024"
    #                     f" --batch_size 1024 --n_hidden 3 --first_dim {first_dims[i]} --seed {seed}")

    # MLP + WD
    for wd in weight_decays:
        commands.append(f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024"
                        f" --batch_size 1024 --n_hidden 3 --first_dim {first_dims[i]} --weight_decay {wd} --seed {10}")

    # LogUniform
    # fourier_dim = lff_configs[i][0]
    # for high in log_uniform_ranges:
    #     for seed in [10, 20]:
    #         commands.append(base_str + f' --network_class LogUniformFourierMLP --fourier_dim {fourier_dim} --sigma {high} --seed {seed}')

    for command in commands:
        count += 1
        print(f'{command} --max_timesteps {max_timesteps} --ec2')
        if count % 10 == 0:
            print(count)
print(count)
# ablation_envs = []
