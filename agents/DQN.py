# Code adapted from Pytorch tutorial for DQN

import gym
import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state','next_action'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class MlpPolicy(nn.Module):

    def __init__(self,state_dim,action_dim,hidden_layers, act = F.relu):
        super(MlpPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc3 = nn.Linear(hidden_layers[1], action_dim)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.act = act
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))        
        return self.fc3(x)
        
class DQN (object):
    
    def __init__(self,env,lr = 1e-3,replay_capacity = 10000):
        
        num_neurons = 256 #Cartpole [64]
        hidden_layers = [num_neurons,num_neurons] 
        state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.policy_net = MlpPolicy(state_dim,self.action_dim,hidden_layers)
        self.target_net = MlpPolicy(state_dim,self.action_dim,hidden_layers)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.replay_buffer = ReplayMemory(replay_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 500
        self.steps_done = 0

        
    def learn(self):
        
        self.gamma = 0.99
        self.batch_size = 32# Cartpole and Highway:32
        
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))
        
        if self.steps_done % 1000 ==0 : # Cartpole: 2000
            self.target_net.load_state_dict(self.policy_net.state_dict())
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute MSE
        loss =F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def select_action(self,state):
        
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done  = self.steps_done + 1
        if sample > eps_threshold:
            with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device)

    def select_greedy_action(self,state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(0)[1].view(1, 1)
    
    def select_eval_action(self,state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(0)[1].view(1, 1)
                
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
