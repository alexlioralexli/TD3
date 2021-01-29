# --expID 27
envs = ['reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1', 'drawer-open-v1', 'drawer-close-v1',
        'button-press-topdown-v1', 'peg-insert-side-v1', 'window-open-v1', 'window-close-v1']
total = 0
for env in envs:
    base_str = f"python main.py --policy PytorchSAC --env mw.{env} --n_hidden 2 --hidden_dim 400 --batch_size 128 --start_timesteps 5000"
    commands = [f"python main.py --policy PytorchSAC --env mw.{env} --n_hidden 3 --hidden_dim 400 --first_dim 1024 --batch_size 128 --start_timesteps 5000"]
    # fourier features
    for type in ['--concatenate_fourier --train_B']:
        for sigma in [0.01, 0.001]:
            commands.append(base_str + f' --network_class FourierMLP --sigma {sigma} --fourier_dim 1024 {type}')
    count = 0
    for command in commands:
        gpus = list(range(10))
        for seed in [10, 20]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'CUDA_VISIBLE_DEVICES={gpus[count]} taskset -c a-b {command} --seed {seed} &')
            count = (count + 1) % len(gpus)
