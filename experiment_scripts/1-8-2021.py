"""
- For both Ant and Humanoid:
- MLP, seeds 10-50
- FourierMLP, sigma=0.01, 0.03, fourier_dim 256, 512, 1024, seeds 10-50
    - concatenate, train_B, concatenate and train_B
"""
exp_strings = []

# normal MLP
exp_strings.append('--expID 0')

# fourier MLP
for type in ['--concatenate_fourier --expID 1',
             '--train_B --expID 2',
             '--concatenate_fourier --train_B --expID 3']:
    for sigma in [0.01, 0.03]:
        for fourier_dim in [256, 512, 1024]:
            exp_strings.append(f'--network_class FourierMLP {type} --sigma {sigma} --fourier_dim {fourier_dim}')

for env in ['Ant-v3', 'Humanoid-v3']:
    for string in exp_strings:
        for seed in [10, 20, 30, 40, 50]:
            print(f'python main.py --env {env} {string} --seed {seed}')
