# fair baseline, 20 runs, cthulhu 5
count = 0
envs = ['Ant-v3', 'HalfCheetah-v3']
fair_first_widths = [869, 882]
for env, first_dim in zip(envs, fair_first_widths):
    for depth in ['--n_hidden 2 --expID 12', '--n_hidden 3 --expID 13']:
        for seed in [10, 20, 30, 40, 50]:
            print(f'CUDA_VISIBLE_DEVICES={count % 5} python main.py --env {env} --first_dim {first_dim} {depth} --seed {seed} &')
            count += 1

