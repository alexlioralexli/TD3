envs = ['Ant-v3', 'HalfCheetah-v3', 'Hopper-v2', 'Walker2d-v2']
env_folders = [['/home/pathak-visitor1/workspace/TD3/logs/Ant-v3-td3-01-14-2021/Ant-v3-td3-FourierMLP-exp16-01-14-2021_18-21-19-563065',
                '/home/pathak-visitor1/workspace/TD3/logs/Ant-v3-td3-01-14-2021/Ant-v3-td3-MLP-exp16-01-14-2021_18-20-58-515211'],
               ['/home/pathak-visitor1/workspace/TD3/logs/HalfCheetah-v3-td3-01-14-2021/HalfCheetah-v3-td3-FourierMLP-exp16-01-14-2021_18-21-19-959718',
                '/home/pathak-visitor1/workspace/TD3/logs/HalfCheetah-v3-td3-01-14-2021/HalfCheetah-v3-td3-MLP-exp16-01-14-2021_18-20-58-517651'],
               ['/home/pathak-visitor1/workspace/TD3/logs/Hopper-v2-td3-01-14-2021/Hopper-v2-td3-FourierMLP-exp16-01-14-2021_18-21-19-915820',
                '/home/pathak-visitor1/workspace/TD3/logs/Hopper-v2-td3-01-14-2021/Hopper-v2-td3-MLP-exp16-01-14-2021_18-20-58-889674'],
               ['/home/pathak-visitor1/workspace/TD3/logs/Walker2d-v2-td3-01-14-2021/Walker2d-v2-td3-FourierMLP-exp16-01-14-2021_18-21-19-565278',
                '/home/pathak-visitor1/workspace/TD3/logs/Walker2d-v2-td3-01-14-2021/Walker2d-v2-td3-MLP-exp16-01-14-2021_18-20-58-732887']]

files = ['itr250000', 'final']
expl_noises = [0, 0.1]

for i, env in enumerate(envs):
    for folder in env_folders[i]:
        for file in files:
            for expl_noise in expl_noises:
                print(f'python make_state_tsne.py --env {env} --n_timesteps 25000 --expl_noise {expl_noise} --load_model {folder + "/" + file}')