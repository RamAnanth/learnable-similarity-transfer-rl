import numpy as np
import random


import gym
import highway_env
import torch

import os 

from agents.DQN import DQN

import pickle

from sklearn.metrics import auc 
from ray import tune
from env.reacher import ReacherDiscreteEnv
from env.highway import highway_env

import argparse

device = torch.device("cpu")

    
def train_source(config):

    lr = config["lr"]
    if 'Reacher' in args.env:
        env = ReacherDiscreteEnv(friction=0)
    elif 'highway' in args.env:
        env = highway_env()
    else:
        env = gym.make(args.env)
    
    env.seed(1003)
    
    agent = DQN(env, lr=lr)
    
    num_episodes = args.num_episodes
    state_dim = env.observation_space.shape[0]
    
    if 'highway' in args.env:
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    
    total_rewards = []

    for i_episode in range(num_episodes):
    # Initialize the environment and state
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and perform an action
            action = agent.select_action(torch.from_numpy(state).float())
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device).float()
            
            total_reward += reward
            if not done:
                next_state = next_state
                memory_next_state = torch.from_numpy(next_state).float().view(1,state_dim)
                next_action = agent.select_action(memory_next_state.float())
            else:
                next_state = None
                memory_next_state = None
                next_action = None
            
            memory_state = torch.from_numpy(state).float().view(1,state_dim)
            
            # Store the transition in memory
            agent.replay_buffer.push(memory_state, action, reward, memory_next_state, next_action)
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the target network)
            agent.learn()
            if done:
                total_rewards.append(total_reward)
                tune.report(episode_return=total_reward.item())
                break
    
    
    auc_score = auc(np.arange(len(total_rewards)),np.array(total_rewards)/len(total_rewards))
    
    # Set the save file path based on the environment
    if 'CartPole' in args.env:
        data_path = '/data_cartpole/dqn_model_cartpole'
    
    elif 'Acrobot' in args.env:
        data_path = '/data_acrobot/dqn_model_acrobot'
        
    elif 'LunarLander' in args.env:
        data_path = '/data_lander/dqn_model_lander'
        
    elif 'Reacher' in args.env:
        data_path = '/data_reacher/dqn_model_reacher'+str(lr)
    
    elif 'highway' in args.env:
        data_path = '/data_highway/dqn_model_highway'+str(lr)
        
                 
    filepath = path + data_path
    torch.save(agent.policy_net.state_dict(),filepath)
        
    tune.report(auc_score = auc_score)
#    return total_rewards, agent

def hyperparam_tune(config_dict, run):
    analysis = tune.run(
    run,
    config=config_dict)

    print("Best config: ", analysis.get_best_config(
    metric = "auc_score", mode="max"))    
    
    print(analysis.results_df)    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Transfer learning for Dynamics Mismatch')
    parser.add_argument('--seed', default = 0, type = int, help = "Random number seed for the experiment")
    parser.add_argument('--num_episodes', default = 3000, type = int, help = "Number of episodes to run for the learning algorithm")    
    parser.add_argument('--env', default = 'CartPole-v0', type = str, help = "Name of the environment to be used")
    args = parser.parse_args()
    
    path = os.getcwd()
    # Set the random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    tune_hyperparams = True
    config_dict = {
              #"lr": tune.grid_search([1e-4])
              "lr": tune.grid_search([1e-4,1e-3,5e-4,5e-3,1e-2]),
              }

    if tune_hyperparams:
        hyperparam_tune(config_dict, train_source)
