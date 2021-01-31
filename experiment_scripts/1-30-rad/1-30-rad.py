envs = ['walker.run', 'acrobot.swingup', 'cheetah.run', 'hopper.hop', 'quadruped.run', 'walker.run']
for env in envs:
    domain, task = env.split('.')
    for seed in [10]:
        print(f"CUDA_VISIBLE_DEVICES=0 taskset -c a-b python train.py --domain_name {domain} --task_name {task} --encoder_type identity --work_dir ./logs/ --action_repeat 1 --num_eval_episodes 10 --agent rad_sac --seed {seed} --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000")