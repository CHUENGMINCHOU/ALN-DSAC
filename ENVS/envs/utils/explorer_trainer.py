import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.distributions.categorical import Categorical
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ENVS.envs.utils.info import *
from ENVS.envs.utils.state import JointState

from scipy import spatial


def average(input_list):
    if input_list:
        return sum(np.array(input_list).tolist()) / len(input_list)
    else:
        return 0

class Explorer(object): # run k epis to update buffer in training; to test the model in eva/testing
    def __init__(self, args, env, robot, device, memory, agent_policy):
        self.env = env
        self.robot = robot
        self.device = device

        self.memory = memory
        self.n_epi_experience = []

        self.batch_size = args.batch_size

        self.total_steps = []
        self.total_rewards = []

        self.args = args
        self.target_update_interval = args.target_update_interval
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.agent_policy = agent_policy
        self.input_dim = agent_policy.input_dim()
        self.action_space = agent_policy.input_dim()
        
        #shared attention network
        self.attention_network = agent_policy.Shared_Attention_Net
        self.attention_optim = agent_policy.attention_optim

        self.policy_network = agent_policy.policy_network
        self.policy_network_optim = agent_policy.policy_network_optim

        self.critic = agent_policy.critic
        self.critic_target = agent_policy.critic_target
        self.critic_optim = agent_policy.critic_optim

        self.automatic_entropy_tuning = agent_policy.automatic_entropy_tuning

        self.target_entropy = agent_policy.target_entropy
        self.log_alpha = agent_policy.log_alpha
        self.alpha_optim = agent_policy.alpha_optim

    def compute_returns(self, rewards, masks):
        R = 0
        returns = []
        for i in reversed(range(len(rewards))):
            #gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
            R = rewards[i] + 0.6 * R * masks[i]
            returns.insert(0, R)
        return returns


    def update_buffer(self, n_epi_experience):
        for epi in n_epi_experience:
            states = epi[0]
            actions_indice = epi[1]
            rewards = epi[2]
            next_states = epi[3]
            masks = epi[4]
            priorities = epi[5]
            for (state_, action_indice_, reward_, next_state_, mask_, priority_) in zip(states, actions_indice, rewards, next_states, masks, priorities):
                self.memory.push(state_, action_indice_.squeeze(0), reward_, next_state_, mask_, priority_)



    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


######################
    def update_parameters_mix_batch(self, memory, states, actions_indice, rewards, next_states, masks):
        for i in range(1):
            online_size = len(actions_indice)

            states = np.array([t.numpy() for t in states])
            next_states = np.array([t.numpy() for t in next_states])
            masks = np.array([m.numpy() for m in masks])

            states = torch.FloatTensor(states)
            actions_indice = torch.FloatTensor(actions_indice).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)

            masks = torch.FloatTensor(masks)
            
            #rest prioritezed batch
            experience_sampled = []
            state_batch, action_indice_batch, reward_batch, next_state_batch, mask_batch, priority_batch = memory.sample(batch_size=self.batch_size)
            for i in range(self.batch_size):
                experience_sampled.append([state_batch[i], action_indice_batch[i], reward_batch[i], next_state_batch[i], mask_batch[i], priority_batch[i]])
            experience_sampled = np.array(experience_sampled)
            index = np.argsort(experience_sampled[:, 5])[::-1]

            experience_sampled_copy = experience_sampled.copy()
            for i in range(len(index)):
                experience_sampled[i] = experience_sampled_copy[index[i]]

            experience_train = experience_sampled[:self.batch_size-online_size]
            state_batch = np.array(experience_train[:, 0].tolist())

            action_indice_batch = np.array(experience_train[:, 1].tolist())
            reward_batch = np.array(experience_train[:, 2].tolist())
            next_state_batch = np.array(experience_train[:, 3].tolist())
            mask_batch = np.array(experience_train[:, 4].tolist())
            
            state_batch = torch.FloatTensor(state_batch)
            state_batch = torch.cat((state_batch, states), 0).to(self.device)

            next_state_batch = torch.FloatTensor(next_state_batch)
            next_state_batch = torch.cat((next_state_batch, next_states), 0).to(self.device)

            action_indice_batch = torch.FloatTensor(action_indice_batch)
            action_indice_batch = torch.cat((action_indice_batch, actions_indice), 0).to(self.device)

            reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
            reward_batch = torch.cat((reward_batch, rewards), 0).to(self.device)

            mask_batch = torch.FloatTensor(mask_batch)

            mask_batch = torch.cat((mask_batch, masks), 0).to(self.device)

            with torch.no_grad():
                #calculate next target Q
            #calculate attention weight
                next_attention_weight = self.attention_network(next_state_batch)
                next_state_action_prob, next_state_log_action_prob, _ = self.policy_network.sample(next_state_batch, next_attention_weight)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_attention_weight)

                min_qf_next_target = (next_state_action_prob * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_action_prob)).sum(dim=1, keepdim=True)
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

            #calculate current Q
            #calculate attention weight
            current_attention_weight = self.attention_network(state_batch)
            qf1, qf2 = self.critic(state_batch, current_attention_weight) 

            qf1 = qf1.gather(1, action_indice_batch.long())
            qf2 = qf2.gather(1, action_indice_batch.long())

            qf1_loss = F.mse_loss(qf1, next_q_value) 
            qf2_loss = F.mse_loss(qf2, next_q_value)  
            qf_loss = qf1_loss + qf2_loss

            #calculate attention weight
            #current_attention_weight = self.attention_network(state_batch)
            pi, log_pi, _ = self.policy_network.sample(state_batch, current_attention_weight)

            with torch.no_grad():
                qf1_pi, qf2_pi = self.critic(state_batch, current_attention_weight)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
            entropies = -torch.sum(pi * log_pi, dim=1, keepdim=True)
            q_exp = torch.sum(torch.min(qf1_pi, qf2_pi) * pi, dim=1, keepdim=True)
            policy_loss = (-q_exp - self.alpha * entropies).mean()


            #backward of attention network
            attention_loss = policy_loss.detach()
            attention_loss = Variable(attention_loss, requires_grad=True)

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            self.policy_network_optim.zero_grad()
            policy_loss.backward()
            self.policy_network_optim.step()
        
            self.attention_optim.zero_grad()
            attention_loss.backward()
            self.attention_optim.step()

        
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (self.target_entropy - entropies).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()

            self.soft_update(self.critic_target, self.critic, self.tau)



    def update_parameters(self, memory, batch_size, updates, episode):
        experience_sampled_1 = []
        experience_sampled_2 = []
        state_batch, action_indice_batch, reward_batch, next_state_batch, mask_batch, priority_batch = memory.sample(batch_size=batch_size)

        for i in range(batch_size):
            experience_sampled_1.append([state_batch[i], action_indice_batch[i], reward_batch[i], next_state_batch[i], mask_batch[i], priority_batch[i]])

        state_batch, action_indice_batch, reward_batch, next_state_batch, mask_batch, priority_batch = memory.sample(batch_size=batch_size)
        for i in range(batch_size):
            experience_sampled_2.append([state_batch[i], action_indice_batch[i], reward_batch[i], next_state_batch[i], mask_batch[i], priority_batch[i]])

        #cosin difference/similarity
        experience_sampled_1_priority = np.array(experience_sampled_1)[:, 5]
        experience_sampled_2_priority = np.array(experience_sampled_2)[:, 5]
        similarity = 1 - spatial.distance.cosine(experience_sampled_1_priority, experience_sampled_2_priority)

        if similarity < 0:
            experience_combined = np.array(experience_sampled_1 + experience_sampled_2)
            index = np.argsort(experience_combined[:, 5])[::-1]
            experience_combined_copy = experience_combined.copy()
            for i in range(len(index)):
                experience_combined[i] = experience_combined_copy[index[i]]

            experience_train = experience_combined[:batch_size]
            state_batch = np.array(experience_train[:, 0].tolist())

            action_indice_batch = np.array(experience_train[:, 1].tolist())
            reward_batch = np.array(experience_train[:, 2].tolist())
            next_state_batch = np.array(experience_train[:, 3].tolist())
            mask_batch = np.array(experience_train[:, 4].tolist())
        else:
            state_batch, action_indice_batch, reward_batch, next_state_batch, mask_batch, priority_batch = memory.sample(batch_size=batch_size)


        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_indice_batch = torch.FloatTensor(action_indice_batch).to(self.device)#.unsqueeze(0)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)#for reward
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)#.unsqueeze(1)

        #compute next Q value 
        with torch.no_grad():
            #calculate attention weight
            next_attention_weight = self.attention_network(next_state_batch)

            #calculate next target Q
            next_state_action_prob, next_state_log_action_prob, _ = self.policy_network.sample(next_state_batch, next_attention_weight)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_attention_weight)
            min_qf_next_target = (next_state_action_prob * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_action_prob)).sum(dim=1, keepdim=True)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        #calculate current Q
        current_attention_weight = self.attention_network(state_batch)
        qf1, qf2 = self.critic(state_batch, current_attention_weight)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = qf1.gather(1, action_indice_batch.long())
        qf2 = qf2.gather(1, action_indice_batch.long())

        #calculate critic loss
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        #calculate policy(actor) loss
        pi, log_pi, _ = self.policy_network.sample(state_batch, current_attention_weight)
        with torch.no_grad():
            qf1_pi, qf2_pi = self.critic(state_batch, current_attention_weight)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
        entropies = -torch.sum(pi * log_pi, dim=1, keepdim=True)
        q_exp = torch.sum(torch.min(qf1_pi, qf2_pi) * pi, dim=1, keepdim=True)
        policy_loss = (-q_exp - self.alpha * entropies).mean()

        #backward of attention network
        attention_loss = policy_loss.detach()
        attention_loss = Variable(attention_loss, requires_grad=True)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        self.policy_network_optim.zero_grad()
        policy_loss.backward()
        self.policy_network_optim.step()
        
        self.attention_optim.zero_grad()
        attention_loss.backward()
        self.attention_optim.step()
        

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (self.target_entropy - entropies).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        if updates % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)
    

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False,  episode=None, print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []

        updates = 0
        episode_steps = 0

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            actions_indice = []
            rewards = []
            next_states = []
            masks = []

            while not done:
                action, action_indice, action_prob, state_joint = self.robot.act(ob)
                state = self.robot.policy.last_state 
                states.append(state)

                #next state version_2
                next_self_state = self.agent_policy.propagate(state_joint.self_state, action)
                ob_next_pred, _, _, _ = self.env.onestep_lookahead(action)
                next_state_joint = JointState(next_self_state, ob_next_pred)
                next_state = self.agent_policy.transform(next_state_joint)
                next_states.append(next_state)

                ob_next, reward, done, info = self.env.step(action)
                actions_indice.append(action_indice)
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
                mask = float(not done)
                masks.append(torch.tensor([1-done], dtype=torch.float32, device=self.device))
                ob = ob_next

                if len(self.memory) > self.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.args.updates_per_step):
                        # Update parameters of all the networks
                        self.update_parameters(self.memory, self.batch_size, updates, episode)
                        updates += 1

                episode_steps += 1

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            # visualize the training data
            #self.env.render('video', self.args.video_file)

            priorities = self.compute_returns(rewards, masks)
            if update_memory == True:
                if episode % self.args.n_epi_store == 0:
                    self.update_buffer(self.n_epi_experience)
                    self.n_epi_experience = []
                    self.n_epi_experience.append([states, actions_indice, rewards, next_states, masks, priorities])
                else:
                    self.n_epi_experience.append([states, actions_indice, rewards, next_states, masks, priorities])




            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            avg_cumulative_rewards = average(cumulative_rewards)

            ##online learning
            if self.args.mix_batch == True and len(self.memory) > self.batch_size:
                for i in range(self.args.count_mix_batch):
                    self.update_parameters_mix_batch(self.memory, states, actions_indice, rewards, next_states, masks)


        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))

        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    
