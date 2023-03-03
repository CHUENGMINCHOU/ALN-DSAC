import argparse
import torch
import logging
import configparser
import gym
import numpy as np
import os
from ENVS.envs.policy.policy_factory import policy_factory
from ENVS.envs.utils.robot import Robot
from ENVS.envs.policy.orca import ORCA
from ENVS.envs.utils.explorer_trainer import *
from ENVS.envs.utils.replay_memory import *




def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='ENVS/envs/configs/env.config')
    parser.add_argument('--policy_config', type=str, default='ENVS/envs/configs/policy.config')
    parser.add_argument('--policy', type=str, default='sa_sac_rl')
    #parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--output_dir', type=str, default='ENVS/data/output')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=True, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')

    parser.add_argument('--update_memory', default=True, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
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
    parser.add_argument('--train_episodes', type=int, default=100000)
    parser.add_argument('--sample_episodes', type=int, default=1)

    parser.add_argument('--mix_batch', default=False, action='store_true')
    parser.add_argument('--count_mix_batch', type=int, default=1)
    parser.add_argument('--n_epi_store', type=int, default=10)
    args = parser.parse_args()
    args = parser.parse_args()


    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    agent_policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    agent_policy.configure(policy_config, device, args)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('sil-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(agent_policy)
    env.set_robot(robot)

    agent_policy.set_phase(args.phase)
    agent_policy.set_device(device)
    agent_policy.set_env(env)
    agent_policy.load_models(30000)
    robot.print_info()

    memory = ReplayMemory(args.replay_size, args.seed)
    explorer = Explorer(args, env, robot, device, memory, agent_policy)

    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action, _, _, _ = robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
