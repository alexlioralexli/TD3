import math

envs = [
    'dm.acrobot.swingup',
    'dm.cheetah.run',
    'dm.finger.turn_hard',
    'dm.walker.run',
    'dm.hopper.hop',
]

lff_configs = [
    (1024, 0.01),
    (64, 0.003),
    (1024, 0.01),
    (1024, 0.001),
    (1024, 0.001),
]

env_dims = [
    (6, 1),
    (17, 6),
    (12, 2),
    (24, 6),
    (15, 4),
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
wd = 1e-5

lr = '1e-4'
count = 0
for i, env in enumerate(envs):
    commands = []
    base_str = f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 --lr {lr}"
    max_timesteps = '1000000'

    # MLP + WD
    for seed in [10, 20, 30]:
        commands.append(f"python main.py --policy PytorchSAC --env {env} --start_timesteps 5000 --hidden_dim 1024"
                        f" --batch_size 1024 --n_hidden 3 --first_dim {first_dims[i]} --weight_decay {wd} --lr {lr} --seed {seed}")

    # LogUniform
    # fourier_dim = lff_configs[i][0]
    # for high in log_uniform_ranges:
    #     for seed in [10, 20]:
    #         commands.append(base_str + f' --network_class LogUniformFourierMLP --fourier_dim {fourier_dim} --sigma {high} --seed {seed}')

    for command in commands:
        count += 1
        print(f'CUDA_VISIBLE_DEVICES=0 taskset -c a-b {command} --max_timesteps {max_timesteps} &')
        if count % 10 == 0:
            print(count)
print(count)
# ablation_envs = []
