"""
An agent to directly deploy the optimal policy learnt from the source task
"""

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
                        ('state', 'action', 'next_state', 'reward','next_action'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(20)

class MlpPolicy(nn.Module):
    # Network that defines Q as an MLP
    def __init__(self,state_dim,action_dim,hidden_layers, act = F.relu):
        super(MlpPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_layers[0])
        torch.nn.init.xavier_normal_(self.fc1.weight) # Xavier Normal Initialization
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        torch.nn.init.xavier_normal_(self.fc2.weight) # Xavier Normal Initialization
        self.fc3 = nn.Linear(hidden_layers[1], action_dim)
        torch.nn.init.xavier_normal_(self.fc3.weight) # Xavier Normal Initialization
        self.act = act

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))        
        return self.fc3(x)
        
class DQN_source(object):
    
    def __init__(self,env, source_path, replay_capacity = 10000):
        
        # Neural Network architecture
        num_neurons = 64
        if env.name in ['Reacher','LunarLander','Acrobot']:
            num_neurons = 256
        hidden_layers = [num_neurons,num_neurons] #Cartpole [64:64]
        state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Intialize Q network and target network
        self.policy_net = MlpPolicy(state_dim,self.action_dim,hidden_layers)
        self.target_net = MlpPolicy(state_dim,self.action_dim,hidden_layers)
        
        self.replay_buffer = ReplayMemory(replay_capacity)
        
        # Load the parameters from the saved source DQN
        self.policy_net.load_state_dict(torch.load(source_path))
        
    def learn(self):
        pass
    
    # Return the action corresponding to source optimal policy    
    def select_action(self,state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(0)[1].view(1, 1)
       
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