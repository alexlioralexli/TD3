"""
SAC ablation Value vs Policy vs Both vs Baseline: 2M * 4envs * 5seeds * 2NewMethods
SAC ablation: 1M * 4envs * 3seeds * 3methods (can leave it as is), just humanoid run 2 methods
"""
import math

envs = [
    'dm.finger.turn_hard',
    'dm.walker.run',
    'dm.quadruped.run',
    'dm.humanoid.run',
]

lff_configs = [
    (1024, 0.01),
    (1024, 0.001),
    (1024, 0.01),
    (1024, 0.01)
]

env_dims = [
    (12, 2),
    (24, 6),
    (78, 12),
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


lr = '1e-4'
count = 0
for i, env in enumerate(envs):
    commands = []
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr {lr}"
    if i != 3:
        max_timesteps = '2000000'
    else:
        max_timesteps = '5000000'

    # LFF
    fourier_dim, sigma = lff_configs[i]
    for seed in [10, 20, 30]:
        commands.append(base_str + f' --network_class FourierMLP --concatenate_fourier --train_B'
                                   f' --sigma {sigma} --fourier_dim {fourier_dim} --mlp_qf --seed {seed}')
        commands.append(base_str + f' --network_class FourierMLP --concatenate_fourier --train_B'
                                   f' --sigma {sigma} --fourier_dim {fourier_dim} --mlp_policy --seed {seed}')

    # Humanoid Run catchup
    if i == 3:
        for seed in [10, 20, 30]:
            commands.append(base_str + f' --network_class FourierMLP --train_B'
                                       f' --sigma {sigma} --fourier_dim {fourier_dim} --seed {seed}')
            commands.append(base_str + f' --network_class FourierMLP --concatenate_fourier'
                                       f' --sigma {sigma} --fourier_dim {fourier_dim} --seed {seed}')

    for command in commands:
        count += 1
        print(f'CUDA_VISIBLE_DEVICES=0 taskset -c a-b {command} --max_timesteps {max_timesteps}')
print(count)
# ablation_envs = []
