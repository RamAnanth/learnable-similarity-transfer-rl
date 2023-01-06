import numpy as np
import random

import gym
import torch

import os 

from agents.DQN_reshaped import DQN_Reshaped
from agents.DQN_source import DQN_source
from agents.DQN import DQN

from env.cartpole_transfer import CartPoleEnv_transfer
from env.acrobot_transfer import AcrobotEnv_transfer
from env.lunarlander_transfer import LunarLanderTransfer
from env.reacher import ReacherDiscreteEnv

import pickle

from sklearn.metrics import auc 
from ray import tune

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

device = torch.device("cpu")

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
    
def test_policy():
    
    if args.env == 'CartPole':
        data_path = '/data_cartpole/dqn_model_cartpole'
    elif args.env == 'Acrobot':
        data_path = '/data_acrobot/dqn_model_acrobot'
    elif args.env == 'LunarLander':
        data_path = '/data_lander/dqn_model_lander'
    elif args.env == 'Reacher':
        data_path = '/data_reacher/dqn_model_reacher'
    
    source_path = path + data_path
    num_episodes = args.eval_episodes
    if args.env == 'CartPole':
        env = CartPoleEnv_transfer(gravity_factor = config["dynamics_factor"])
    elif args.env == 'Acrobot':
        env = AcrobotEnv_transfer(link_mass = config["dynamics_factor"])
    elif args.env == 'LunarLander':
        env = LunarLanderTransfer(density_ratio = config["dynamics_factor"])
    elif args.env == 'Reacher':
        env = ReacherDiscreteEnv(friction = args.dynamics_factor)
    
    num_neurons = 256
    hidden_layers = [num_neurons,num_neurons] #Cartpole [64:64]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = MlpPolicy(state_dim,action_dim,hidden_layers)
    policy_net.load_state_dict(torch.load(source_path))
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            with torch.no_grad():
                action = policy_net(torch.from_numpy(obs).float()).max(0)[1].view(1, 1)
            obs,_,done,_ = env.step(action.item())
            
    env.close()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing and visualizing agent performance')
    parser.add_argument('--eval_episodes', default = 50, type = int, help = "Number of episodes to run for the evaluation algorithm")    
    parser.add_argument('--env', default = 'CartPole', type = str, help = "Name of the environment to be used to run experiments. Currently supports choices from ['CartPole','Acrobot','LunarLander']")
    parser.add_argument('--dynamics_factor', default = 1, type = float, help = "Ratio of target task dynamics parameter to source task dynamics parameter")
    
    path = os.getcwd()
    args = parser.parse_args()
    
    if args.env not in ['CartPole','Acrobot','LunarLander', 'Reacher']:
        raise Exception('Environment {} currently not supported. Please choose among currently supported environments ["CartPole","Acrobot","LunarLander", "Reacher"]'.format(args.env))
        
    test_policy()