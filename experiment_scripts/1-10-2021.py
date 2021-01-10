"""
- For Half Cheetah:
- try deeper MLPs
- FourierMLP: try different sigma values
- MLP, seeds 10-50, with 1 hidden and 2 hidden
- FourierMLP, sigma=0.01, 0.03, fourier_dim 256, 512, 1024, seeds 10-50
    - concatenate, train_B, concatenate and train_B
- 200 total experiments
"""
exp_strings = []

# normal MLP
exp_strings.append('--n_hidden 3 --expID 9')

# fourier MLP
for type in ['--train_B --expID 10',
             '--concatenate_fourier --train_B --expID 11']:
    for n_hidden in [1, 2]:
        for sigma in [0.0003, 0.001, 0.003, 0.006]:
            exp_strings.append(f'--network_class FourierMLP {type} --n_hidden {n_hidden} --sigma {sigma} --fourier_dim 1024')

for env in ['HalfCheetah-v3']:
    for string in exp_strings:
        for seed in [10, 20, 30, 40, 50]:
            print(f'python main.py --env {env} {string} --seed {seed}')
