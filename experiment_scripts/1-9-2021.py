"""
- For both Ant and Half Cheetah:
- MLP, seeds 10-50, with 1 hidden and 2 hidden
- FourierMLP, sigma=0.01, 0.03, fourier_dim 256, 512, 1024, seeds 10-50
    - concatenate, train_B, concatenate and train_B
- 200 total experiments
"""
exp_strings = []

# normal MLP
exp_strings.append('--n_hidden 1 --expID 4')
exp_strings.append('--n_hidden 2 --expID 5')

# fourier MLP
for type in ['--concatenate_fourier --expID 6',
             '--train_B --expID 7',
             '--concatenate_fourier --train_B --expID 8']:
    for sigma in [0.01, 0.03]:
        for fourier_dim in [256, 512, 1024]:
            exp_strings.append(f'--network_class FourierMLP {type} --sigma {sigma} --fourier_dim {fourier_dim}')

for env in ['Ant-v3', 'HalfCheetah-v3']:
    for string in exp_strings:
        for seed in [10, 20, 30, 40, 50]:
            print(f'python main.py --env {env} {string} --seed {seed}')
