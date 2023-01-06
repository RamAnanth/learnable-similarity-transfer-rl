import numpy as np
import random

import gym
import torch

import os 

from agents.DQN_reshaped import DQN_Reshaped
from agents.DQN_source import DQN_source
from agents.DQN import DQN
from agents.DQN_advice import DQN_Advice

from env.cartpole_transfer import CartPoleEnv_transfer
from env.acrobot_transfer import AcrobotEnv_transfer
from env.lunarlander_transfer import LunarLanderTransfer
from env.reacher import ReacherDiscreteEnv

import pickle

from sklearn.metrics import auc 
from ray import tune

import argparse

device = torch.device("cpu")

def save_results(result_data, with_transfer = False, transfer_mode = 'delta'):
    filename = '_with_transfer'+'_'+transfer_mode if with_transfer else '_without_transfer'
    result_file = 'results/DQN'+filename
    method_name = 'DQN'+filename
    with open(result_file, 'wb') as f:
        pickle.dump((result_data, method_name), f)
    
def run_pies(config):
    target_task = config["target_task"]
    with_transfer = config["with_transfer"]
    lr = config["lr"]
    seed = config["seed"]
    transfer_mode = config["transfer_mode"]

# Set the random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    eval_interval = 100
    
    # Choose the class corresponding to the environment
    if target_task :
        if args.env == 'CartPole':
            env = CartPoleEnv_transfer(dynamics_factor = config["dynamics_factor"])
        elif args.env == 'Acrobot':
            env = AcrobotEnv_transfer(moi = config["dynamics_factor"])
        elif args.env == 'LunarLander':
            env = LunarLanderTransfer(density_ratio = config["dynamics_factor"])
        elif args.env == 'Reacher':
            env = ReacherDiscreteEnv(friction = config["dynamics_factor"])
    else:
        env = gym.make('CartPole-v0')
    
    env.seed(args.seed)
    
    if args.env == 'CartPole':
        data_path = '/data_cartpole/dqn_model_cartpole'
    elif args.env == 'Acrobot':
        data_path = '/data_acrobot/dqn_model_acrobot'
    elif args.env == 'LunarLander':
        data_path = '/data_lander/dqn_model_lander'
    elif args.env == 'Reacher':
        data_path = '/data_reacher/dqn_model_reacher'
    
    source_path = path + data_path
    # Get number of actions from gym action space
    if with_transfer:
        if 'advice' in args.expt_type:
            agent = DQN_Advice(env,source_path, lr=lr, transfer_mode = transfer_mode)
        else:
            agent = DQN_Reshaped(env,source_path, lr=lr, transfer_mode = transfer_mode)
            if transfer_mode =='source':
                print('Reusing source')
                agent = DQN_source(env, source_path)
    else:
        print(transfer_mode)
        agent = DQN(env, lr=lr)
    
    num_episodes = args.num_episodes
    state_dim = env.observation_space.shape[0]
    
    total_rewards = []
    eval_mean_returns = []
    
    for i_episode in range(num_episodes):
    # Initialize the environment and state
        total_reward = 0
        done = False
        
        if i_episode > 0 and i_episode%eval_interval == 0:
            num_eval_episodes = 50
            total_eval_rewards = []
    
            for _ in range(num_eval_episodes):
                state_eval = env.reset()
                eval_reward = 0
                done_eval = False
        
                while not done_eval:
                # Select greedy action
                    action = agent.select_eval_action(torch.from_numpy(state_eval).float())
                    state_eval, reward_eval, done_eval, _ = env.step(action.item())
                    eval_reward += reward_eval
        
                total_eval_rewards.append(eval_reward)
            
            eval_mean, _ = np.mean(total_eval_rewards), np.std(total_eval_rewards)/np.sqrt(len(total_eval_rewards))
            eval_mean_returns.append(eval_mean)
            tune.report(eval_return = eval_mean)
        
        state = env.reset()
    
        while not done:
            action = agent.select_action(torch.from_numpy(state).float())
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device).float()
            total_reward += reward
            if not done:
                next_state = next_state
                memory_next_state = torch.from_numpy(next_state).float().view(1,state_dim)
                next_action = agent.select_greedy_action(torch.from_numpy(next_state).float())
            else:
                next_state = None
                memory_next_state = None
                next_action = None
            
            memory_state = torch.from_numpy(state).float().view(1,state_dim)
            
            transition_tuple = (memory_state, action,reward, memory_next_state, next_action)
            
            # Learn the dynamic potential function
            if with_transfer and transfer_mode not in ['source', 'static']:
                agent.learn_phi(transition_tuple)
            
            # Store the transition in memory
            agent.replay_buffer.push(memory_state, action, reward, memory_next_state, next_action)
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization
            agent.learn()
            if done:
                total_rewards.append(total_reward)
                tune.report(episode_return=total_reward.item())
                break
    
    auc_score_eval = auc(np.arange(len(eval_mean_returns)), np.array(eval_mean_returns)/len(eval_mean_returns))
    print(eval_mean_returns)
    if not target_task:
        filepath = path + '/data_cartpole/dqn_model_cartpole'+str(lr)
        torch.save(agent.policy_net.state_dict(),filepath)
        
    tune.report(auc_score = auc_score_eval)

def hyperparam_tune(config_dict, run):
    analysis = tune.run(
    run,
    config=config_dict)

    print("Best config: ", analysis.get_best_config(
    metric = "auc_score", mode="max"))    
    
    print(analysis.results_df)    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Transfer learning for Dynamics Mismatch using Explicit Shaping')
    parser.add_argument('--seed', default = 0, type = int, help = "Random number seed for the experiment")
    parser.add_argument('--num_episodes', default = 4000, type = int, help = "Number of episodes to run for the learning algorithm")    
    parser.add_argument('--env', default = 'CartPole', type = str, help = "Name of the environment to be used to run experiments. Currently supports choices from ['CartPole','Acrobot','LunarLander']")
    parser.add_argument('--dynamics_factor', default = 1, type = float, help = "Ratio of target task dynamics parameter to source task dynamics parameter")
    parser.add_argument('--expt_type', default = 'similarity', type = str, help = "Type of experiment to run. Either similarity based experiments or experiments to compare choice of advice")
    
    
    path = os.getcwd()
    args = parser.parse_args()
    # Set the random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.env not in ['CartPole','Acrobot','LunarLander', 'Reacher']:
        raise Exception('Environment {} currently not supported. Please choose among currently supported environments ["CartPole","Acrobot","LunarLander", "Reacher"]'.format(args.env))
 
 
    tune_hyperparams = True    
    config_dict = {"target_task": True,
                   "dynamics_factor": args.dynamics_factor, # Cartpole: Gravity, Acrobot : Link mass, LunarLander : Lander Mass, Reacher: Y Axis force
              "with_transfer":tune.grid_search([True]),
              "transfer_mode":tune.grid_search(['no_transfer','static', 'static_delta']),
              #"transfer_mode": tune.grid_search(['no_transfer', 'policy', 'advantage', 'qvalue', 'delta_action']),
              "expt_type":args.expt_type,
              "seed":args.seed,
              # "no_transfer" corresponds to DQN, "static_delta" corresponds to delta based explicit shaping with static advice, 
              #"static" corresponds to DQN with static advice using explicit shaping
              "lr": tune.grid_search([1e-4, 5e-4, 5e-3, 1e-3]),
              }
    
    if tune_hyperparams:
        hyperparam_tune(config_dict, run_pies)
   
