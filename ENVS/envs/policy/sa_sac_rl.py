from torch.nn.functional import softmax
import logging
from ENVS.envs.policy.policy import Policy
import numpy as np
from ENVS.envs.utils.utils import *
import torch
import torch.nn as nn
import itertools
from torch.distributions.categorical import Categorical
from ENVS.envs.utils.state import ObservableState, FullState
from ENVS.envs.utils.action import *
import torch.nn.functional as F

import os
from torch.distributions import Normal
from torch.optim import Adam

def build_action_space():
    speeds = [(np.exp((i + 1) / 5) - 1) / (np.e - 1) * 1 for i in range(5)]
    rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    action_space = [ActionXY(0, 0)]
    for rotation, speed in itertools.product(rotations, speeds):
        action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
    return action_space


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

###########################
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Shared_Attention_Net(nn.Module):
    def __init__(self, input_dim, mlp1_dims, mlp2_dims, attention_dims, with_global_state):
        super(Shared_Attention_Net, self).__init__()

        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.apply(weights_init_)

    def forward(self, state):
        size = state.shape
        dim = state.dim()
        if dim == 2:
            state = state.unsqueeze(0)
            size = state.shape
        
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        # masked softmax
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        return self.attention_weights


class QNetwork(nn.Module):
    def __init__(self, input_dim, lstm_input_dim, self_state_dim, lstm_hidden_dim, actions_dim, hidden_dim, cell_size, cell_num, mlp1_dims, mlp2_dims, attention_dims, with_global_state):
        super(QNetwork, self).__init__()

        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, batch_first=True)

        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)

        self.cell_size = cell_size
        self.cell_num = cell_num
        # Q1 architecture
        self.linear1 = nn.Linear(self_state_dim + lstm_hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, actions_dim)

        # Q2 architecture
        self.linear4 = nn.Linear(self_state_dim + lstm_hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, actions_dim)

        self.apply(weights_init_)


    def forward(self, state, attention_weights):
        size = state.shape
        dim = state.dim()
        if dim == 2:
            state = state.unsqueeze(0)
            size = state.shape
      
        self_state = state[:, 0, :self.self_state_dim]

        index = np.argsort(attention_weights)[::-1]
        state_clone = state.clone()
        for i in range(len(index)):
            state[:, i] = state_clone[:, index[i]]

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)

        x1 = F.relu(self.linear1(joint_state))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)

        x2 = F.relu(self.linear4(joint_state))
        x2 = F.relu(self.linear5(x2))
        q2 = self.linear6(x2)

        return q1, q2 



class SaSacRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'SaSacRL_lstm_aw_NPER'
        self.kinematics = 'holonomic'
        self.multiagent_training = 'False'
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def set_common_parameters(self, config):
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def configure(self, config, device, args):
        global_state_dim = config.getint('sa_sac_rl', 'global_state_dim')
        with_interaction_module = config.getboolean('sa_sac_rl', 'with_interaction_module')
        logging.info('Policy: {}LSTM-SASACRL {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))

        self.set_common_parameters(config)
        self.device = device
        self.args = args
        mlp1_dims = [int(x) for x in config.get('sa_sac_rl', 'mlp1_dims').split(', ')]
        self.mlp2_dims = [int(x) for x in config.get('sa_sac_rl', 'mlp2_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sa_sac_rl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sa_sac_rl', 'with_om')
        with_global_state = config.getboolean('sa_sac_rl', 'with_global_state')
        self.multiagent_training = config.getboolean('sa_sac_rl', 'multiagent_training')
        self.action_space = build_action_space()

        #shared attention net
        self.Shared_Attention_Net = Shared_Attention_Net(self.input_dim(), mlp1_dims, self.mlp2_dims, attention_dims, with_global_state)

        self.critic = QNetwork(self.input_dim(), self.lstm_input_dim(), self.self_state_dim, global_state_dim, len(self.action_space), self.args.hidden_size, self.cell_size, self.cell_num, mlp1_dims, self.mlp2_dims, attention_dims, with_global_state).to(device=self.device)
        self.critic_target = QNetwork(self.input_dim(), self.lstm_input_dim(), self.self_state_dim, global_state_dim, len(self.action_space), self.args.hidden_size, self.cell_size, self.cell_num, mlp1_dims, self.mlp2_dims, attention_dims, with_global_state).to(device=self.device)

        hard_update(self.critic_target, self.critic)

        self.policy_network_type = "Categorical"
        self.automatic_entropy_tuning = self.args.automatic_entropy_tuning
        #optimizer of attention net
        self.attention_optim = Adam(self.Shared_Attention_Net.parameters(), lr=self.args.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.args.lr)

        if self.policy_network_type == "Categorical":
            if True:
                self.target_entropy = -torch.prod(torch.Tensor(np.array(self.action_space).shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.args.lr)

            self.policy_network = PolicyNetwork(self.input_dim(), self.lstm_input_dim(), self.self_state_dim, global_state_dim, len(self.action_space), self.args.hidden_size, self.cell_size, self.cell_num, mlp1_dims, self.mlp2_dims, attention_dims, with_global_state).to(device=self.device)
            self.policy_network_optim = Adam(self.policy_network.parameters(), lr=self.args.lr)

        if self.with_om:
            self.name = 'OM-SA_SAC_RL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def lstm_input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def find_action_indice(self, actions, possible_actions):
        possible_actions = np.array(possible_actions)
        actions = torch.tensor(actions).unsqueeze(0)
        indices = np.zeros((actions.shape[0], ), dtype=np.int)

        actions_x = actions[:,0]
        actions_y = actions[:,1]
        possible_actions_x = possible_actions[:,0]
        possible_actions_y = possible_actions[:,1]
        diff_x = actions_x[:,np.newaxis] - possible_actions_x
        diff_y = actions_y[:,np.newaxis] - possible_actions_y
        dist_sq = diff_x ** 2 + diff_y ** 2
        indices = np.argmin(dist_sq, axis=1)
        indices = torch.tensor(indices).squeeze(0)
        return indices

    def predict(self, state, evaluate=False): #joint state
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        
        if self.reach_destination(state):
            action = ActionXY(0, 0)
            action_indice = self.find_action_indice(action, self.action_space)
            return action, action_indice
        
        state_tensor = self.transform(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            #add attention network
            attention_weight = self.Shared_Attention_Net(state_tensor)
            action_probs, _, mean = self.policy_network.sample(state_tensor, attention_weight)
        else:
            _, _, action_probs = self.policy_network.sample(state_tensor)

        action_dist = Categorical(F.softmax(mean, dim=-1))
        action_indice = action_dist.sample().view(-1, 1)
        action = self.action_space[action_indice]

        action_probs = action_probs.squeeze(0).tolist()

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return action, action_indice, action_probs


    def transform(self, state): #joint states
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device) for human_state in state.human_states], dim=0)#pair-wise state tensor
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps], dim=1)
            state_tensor = state_tensor.to(self.device)
        else:
            state_tensor = self.rotate(state_tensor)#robot-centred state tensor

        return state_tensor


    # Save model parameters with attention network
    def save_models(self, episode_count):
        torch.save(self.policy_network.state_dict(), self.args.output_dir + '/SAC_model/' +str(episode_count)+ '_lstm_aw_policy_net.pth')
        torch.save(self.critic.state_dict(), self.args.output_dir + '/SAC_model/' +str(episode_count)+ '_lstm_aw_value_net.pth')
        torch.save(self.Shared_Attention_Net.state_dict(), self.args.output_dir + '/SAC_model/' +str(episode_count)+ '_lstm_attention_net.pth')
    
    # Load model parameters with attention weight
    def load_models(self, episode):
        self.policy_network.load_state_dict(torch.load(self.args.output_dir + '/SAC_model/' +str(episode)+ '_lstm_aw_policy_net.pth'))
        self.critic.load_state_dict(torch.load(self.args.output_dir + '/SAC_model/' +str(episode)+ '_lstm_aw_value_net.pth'))
        self.Shared_Attention_Net.load_state_dict(torch.load(self.args.output_dir + '/SAC_model/' +str(episode)+ '_lstm_attention_net.pth'))
        hard_update(self.critic_target, self.critic)

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                       state.gx, state.gy, state.v_pref, state.theta)
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                       state.v_pref, next_theta)
        else:
            raise ValueError('Type error')

        return next_state
    
    def rotate(self, state):
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]

        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

        return new_state


    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()


