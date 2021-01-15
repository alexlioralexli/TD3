import numpy as np
import torch
import gym
import argparse
import os
import utils
import TD3
import OurDDPG
import DDPG
from models.mlp import MLP, FourierMLP
from logging_utils import save_kwargs, create_env_folder
import os.path as osp
from rlkit_logging import logger
from utils import make_env

NETWORK_CLASSES = dict(
    MLP=MLP,
    FourierMLP=FourierMLP
)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = make_env(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    # network kwargs
    parser.add_argument("--network_class", default="MLP", choices=['MLP', 'FourierMLP'])
    parser.add_argument("--n_hidden", default=1, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--first_dim", default=0, type=int)
    parser.add_argument("--fourier_dim", default=256, type=int)
    parser.add_argument("--sigma", default=1.0, type=float)
    parser.add_argument("--concatenate_fourier", action='store_true')
    parser.add_argument("--train_B", action='store_true')

    # other
    parser.add_argument("--expID", default=9999, type=int)
    parser.add_argument("--test", '-t', action='store_true')
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = make_env(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau
    }

    # custom network kwargs
    kwargs['network_class'] = NETWORK_CLASSES[args.network_class]
    basic_network_kwargs = dict(n_hidden=args.n_hidden,
                                hidden_dim=args.hidden_dim,
                                first_dim=args.first_dim)
    full_network_kwargs = dict(n_hidden=args.n_hidden,
                               hidden_dim=args.hidden_dim,
                               fourier_dim=args.fourier_dim,
                               sigma=args.sigma,
                               concatenate_fourier=args.concatenate_fourier,
                               train_B=args.train_B)
    if args.network_class == 'MLP':
        kwargs['network_kwargs'] = basic_network_kwargs
    else:
        kwargs['network_kwargs'] = full_network_kwargs

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    else:
        raise NotImplementedError

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    # change the kwargs for logging and plotting purposes
    if args.network_class == 'MLP':
        kwargs['network_kwargs'] = full_network_kwargs
    kwargs['expID'] = args.expID
    kwargs['seed'] = args.seed
    kwargs['first_dim'] = max(args.hidden_dim, args.first_dim)

    # set up logging
    log_dir = create_env_folder(args.env, args.expID, 'td3', args.network_class, test=args.test)
    save_kwargs(kwargs, log_dir)
    tabular_log_path = osp.join(log_dir, 'progress.csv')
    text_log_path = osp.join(log_dir, 'debug.log')

    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path)
    exp_name = f'{args.env}-td3-exp{args.expID}'
    logger.push_prefix("[%s] " % exp_name)
    policy.save(osp.join(log_dir, f'itr0'))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))

            logger.record_tabular('Timestep', t)
            logger.record_tabular('Eval returns', evaluations[-1])
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            np.save(osp.join(log_dir, 'evaluations.npy'), evaluations)
            if (t+1) % 50 == 0:
                policy.save(osp.join(log_dir, f'itr{t+1}'))
    policy.save(osp.join(log_dir, f'final'))  # might be unnecessary if everything divides out properly
