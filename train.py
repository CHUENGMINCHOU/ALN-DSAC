import os
import sys
import argparse
import shutil
import logging
import git
import torch
import configparser
import gym
from ENVS.envs.policy.policy_factory import policy_factory
from ENVS.envs.utils.robot import Robot
from ENVS.envs.utils.explorer_trainer import *
from ENVS.envs.utils.replay_memory import *


def main():
    parser = argparse.ArgumentParser('parse config file')
    parser.add_argument('--env_config', type=str, default='ENVS/envs/configs/env.config')
    parser.add_argument('--output_dir', type=str, default='ENVS/data/output')

    parser.add_argument('--update_memory', default=True, action='store_true')
    parser.add_argument('--policy_config', type=str, default='ENVS/envs/configs/policy.config')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--policy', type=str, default='sa_sac_rl')
    parser.add_argument('--train_config', type=str, default='ENVS/envs/configs/train.config')

    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--updates_per_step', type=int, default=1)

    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--train_episodes', type=int, default=40000)
    parser.add_argument('--sample_episodes', type=int, default=1)

    parser.add_argument('--mix_batch', default=True, action='store_true')
    parser.add_argument('--count_mix_batch', type=int, default=1)
    parser.add_argument('--n_epi_store', type=int, default=10)

    args = parser.parse_args()


    #config logging (file, stdout, device)
    log_file = os.path.join(args.output_dir, 'output_lstm_aw.log')
    mode = 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    logging.info('using device: %s', device)

    #config policy
    agent_policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    agent_policy.configure(policy_config, device, args)

    #config environment (env, sim, reward, human, AND robot)
    env_parser = configparser.RawConfigParser()
    env_parser.read(args.env_config)
    env = gym.make('sil-v0')
    env.configure(env_parser)
    robot = Robot(env_parser, 'robot')
    env.set_robot(robot)

    
    #config training paras
    if args.train_config is None:
        parser.error('Train config has to be specified for a trainable network')
    train_parser = configparser.RawConfigParser()
    train_parser.read(args.train_config)
    checkpoint_interval = train_parser.getint('train', 'checkpoint_interval')
    memory = ReplayMemory(args.replay_size, args.seed)

    agent_policy.set_env(env)
    robot.set_policy(agent_policy) 
    robot.print_info()

    #config trainer and explorer
    explorer = Explorer(args, env, robot, device, memory, agent_policy)
    
    episode = 0
    while episode < args.train_episodes:
        # update buffer with 1 episode
        explorer.run_k_episodes(args.sample_episodes, 'train', update_memory=args.update_memory, episode=episode)

        episode += 1

        if episode != 0 and episode % checkpoint_interval == 0:
            agent_policy.save_models(episode)
            logging.info('Actor and Critic models are saved.')

if __name__ == '__main__':
    main()
    






