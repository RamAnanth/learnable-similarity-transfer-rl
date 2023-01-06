import numpy as np
import random
import os

from env.gridworld import GridWorldEnv
from env.inventory import InventoryEnv
#from env.highway import highway

from agents.qlearning import Qlearning
from agents.ppr_qlearning import PPR_Qlearning
from agents.source_reuse import Source
from agents.ttql import TTQlearning
from agents.qlearning_PBRS_arbitrary_rewards import Qlearning_transfer_PBRS_arbitrary_reward
from agents.qlearning_static import Qlearning_transfer_PBRS_static
from agents.qlearning_delta_similarity import Qlearning_Delta
from agents.qlearning_delta_multi import Qlearning_Multi_Delta


import pickle
import argparse

import matplotlib.pyplot as plt
from sklearn.metrics import auc 

# Q-learning
# ver1: update per transition


def run(alpha = 0.5,beta = 0.1, target_task = False, with_transfer = False, alpha2 = 0.1, similarity = ''): # alpha2 = 0.1

    alpha = alpha
    gamma = 0.99
    epsilon = 0.1
    
    if target_task and with_transfer:
        
        # # 20 by 20 gridworld
        if args.env == 'gridworld':
            source_policies = [np.load('data/qlearning_source_20by20.npy')]
            source_values = [np.load('data/qlearning_source_value_20by20.npy')]
        
        elif args.env == 'scm':
            source_policies = [np.load('data/qlearning_source_inventory.npy')]
            source_values = [np.load('data/qlearning_source_value_inventory.npy')]
        
        beta = beta # 0.1
        alpha_2 = alpha2
        
        delta = None
        if args.agent == 'source':
            # Directly deploy source optimal policy
            agent = Source(env, source_values[0])
        elif args.agent == 'static':
            # Q learning with Potential based reward shaping using static potential functions
            agent = Qlearning_transfer_PBRS_static(env,alpha,beta,source_policies, source_values, transfer_mode = transfer_mode)
        elif args.agent == 'delta':
            # Q learning with Delta as a Criterion for Direct Transfer
            agent = Qlearning_Delta(env,alpha,beta, source_policies[0], source_values[0])
        elif args.agent == 'ppr':
            # Q Learning with pi-reuse strategy
            agent = PPR_Qlearning(env,alpha,source_policies[0])
        else :
            # Q learning with chosen potential function
            agent = Qlearning_transfer_PBRS_arbitrary_reward(env, alpha, beta, alpha_2, source_policies[0], source_values[0], similarity = similarity)
        """
        elif args.agent == 'multi_delta':
            if args.env == 'gridworld':
                source_policies.extend(( np.load('data/qlearning_source_20by20.npy') , np.load('data/qlearning_source_20by20.npy')))
                source_values.extend(( np.load('data/qlearning_source_value_20by20.npy') , np.load('data/qlearning_source_value_20by20.npy')))
            agent = Qlearning_Multi_Delta (env, alpha, beta, source_policies, source_values)
        """
    else:
        agent = Qlearning(env, gamma, alpha)

    if args.env == 'gridworld':
        num_episodes = 5000
    elif args.env == 'scm':
        num_episodes = 1000

    episode_returns = []
    # 2. Train
    advice = 0
    counts = []
    
    for i in range(num_episodes):
        
        episode_return = 0
        episode_reshaped_reward = 0
        obs = env.reset()
        iter = 0
        while True: # training

            action = agent.eps_greedy_action(obs,epsilon)
            
            new_obs,reward,done,info = env.step(action)
            
            episode_return = episode_return + reward

            transition_tuple = obs,action,reward,new_obs,done
            
            #episode_reshaped_reward = episode_reshaped_reward + agent.reshaped_reward()
            agent.train(transition_tuple, iter)
            obs = new_obs
            
            if done:
                epsilon = max(epsilon-5e-4,0.05)
                episode_returns.append(episode_return)
                #counts.append(agent.getCountPercentage())
                #print("Episode: {0}, Count Percentage {1}".format(i + 1, agent.getCountPercentage()))
                
                break
            iter += 1

    #Calculate AUC scores
    auc_score = auc(np.arange(num_episodes),np.array(episode_returns)/num_episodes)
    
    param_dict = dict()
    param_dict['alpha'] = alpha
    param_dict['beta'] = beta
    param_dict['alpha2'] = alpha2
    
    """
    if with_transfer:
        np.save('data/converged_delta_20by20.npy', agent.getDelta())
        param_dict['beta'] = beta
    """
    if not target_task:
        result_data =[agent.getPolicy(),agent.getQvalues()]

    if target_task:
        result_data = [agent.getPolicy(), episode_returns]
        if args.agent == 'delta':
            result_data.append(agent.getDelta())
            print(result_data[2].shape)
    
    #print('Counts are', counts)
    print('AUC score is ', auc_score)

    return auc_score, param_dict, result_data
# 3. Save results

def save_results(target_task, result_data, similarity ='' , with_transfer = False):
    # Saves results
    if not target_task:
        if args.env == 'scm':
            np.save('data/qlearning_source_inventory.npy', result_data[0])
            np.save('data/qlearning_source_value_inventory.npy', result_data[1])
        elif args.env == 'gridworld':
            np.save('data/qlearning_source_20by20_1.npy', result_data[0])
            np.save('data/qlearning_source_value_20by20_1.npy', result_data[1])

    if target_task:        
        filename = '_with_transfer_using '+similarity if with_transfer else '_without_transfer'
        method_name = 'Q-learning with Similarity using '+ similarity if with_transfer else 'Q-learning'
        if args.agent=='ppr':
            filename = '_with_transfer_using PPR'
            method_name = 'Q-learning with Probabilistic Policy Reuse'
        elif args.agent=='source':
            filename = '_with_source_reuse'
            method_name = 'Directly Using Source Optimal Policy'
        result_file = args.result_path+'qlearning'+filename

        with open(result_file, 'wb') as f:
            pickle.dump((result_data[1], method_name), f)
            np.save(args.result_path+similarity+'_without_bias.npy', result_data[0])
        if args.agent == 'delta':
            np.save(args.result_path+'delta.npy', result_data[2])
                
def tune(with_transfer, params_range = None, similarity = ''):
    # Function to tune hyperparameters
    max_score = -200
    if not with_transfer:
        for alpha in params_range:
            score, param_dict, result_data = run(alpha = alpha, target_task = True)
            print("AUC score for {0}  is {1} ".format(param_dict,score))
            if score > max_score:
                max_score = score
                best_params = param_dict
                save_results(True,result_data, similarity, with_transfer)

    else:
        for alpha in params_range[:2]:
          for beta in params_range:
              score, param_dict, result_data = run(alpha = alpha, beta = beta, target_task = True ,with_transfer = True, similarity = similarity)
                # score, param_dict, result_data = run(alpha2 = alpha, target_task = True ,with_transfer = True)
              print("AUC score for {0}  is {1} ".format(param_dict,score))
            
              if score > max_score:
                  max_score = score
                  best_params = param_dict
                  save_results(True,result_data, similarity, with_transfer)

    print("Best Hyperparameters are", best_params)
    print("Highest AUC Score is", max_score)
    return max_score

def eval_target_policy():
    # Evaluate policy in the target task
    policy = np.load('results/delta_inventory/delta_without_bias.npy')
    episode_return = 0
    env = InventoryEnv(mean_demand = 5)
    obs = env.reset()
    while True: # testing
        action = policy[obs]
        new_obs,reward,done,info = env.step(action)   
        episode_return = episode_return + reward            
        obs = new_obs
           
        if done:
            print("Episode Return", episode_return)
            break

#eval_target_policy()


      
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer learning for Dynamics Mismatch in the tabular setting')
    # TODO: 
    # Add option to pass some of the parameters and agent through command line
    parser.add_argument('--agent', help='Input the name of the agents you want to train: Choose one from [source, static, delta, ppr, qlearning]',default='qlearning') 
    # Possible values are
    # 'source': Directly deploy source optimal policy
    # 'static': Q learning with Potential based reward shaping using static potential functions
    # 'delta':  Q learning with Delta as a Criterion for Direct Transfer
    # 'ppr': Q Learning with pi-reuse strategy
    # Any other string/default: Q learning with explicit shaping
    parser.add_argument('--seed',type = int ,help='Random seed to be used',default=1000)
    parser.add_argument('--source_task', action='store_true', help='Whether to train source task')
    parser.add_argument('--without_transfer', action='store_true', help='Whether to use base RL algorithm or transfer learning algorithm')
    parser.add_argument('--env', help='Input the name of the environment you want to use: Choose one from [gridworld, scm]',default='gridworld')
    parser.add_argument('--result_path', help='Path to directory where the results are to be stored',default='results/delta_inventory/')    
    
    args = parser.parse_args()
    target_task = not args.source_task
    with_transfer = not args.without_transfer
    result_path = args.result_path
    
    if not os.path.exists(result_path):
        print('Creating the results directory since it does not already exist')
        os.makedirs(result_path)
    
    if args.env == 'gridworld':
      
        #wind = np.array([0,0,0,1,1,1,2,2,1,0,0,0,0,1,1,1,2,2,1,0]) # Source task

        #wind = np.array([0,0,-1,1,1,1,2,2,1,0,0,0,0,1,1,1,2,2,1,0]) # Target task for Scenario 1
        wind = np.array([0,0,0,-1,-1,-1,-2,-2,-1,0,0,0,0,-1,-1,-1,-2,-2,-1,0])  # Target task for Scenario 2

        # Target tasks for empirically evaluating if delta measures some notion of performance similarity across tasks
        #wind = np.array([0,1,1,1,2,1,-1,1,-1,0,0,1,1,1,2,1,-1,1,-1,0]) # moderate change1 20*20
        #wind = np.array([0,1,-1,1,1,1,-2,2,1,0, 0,1,-1,1,1,1,-2,2,1,0]) # moderate change2 20*20

        env = GridWorldEnv(20,20,wind=wind)

    elif args.env == 'scm':
      
        #mean_demand = 2.5 # Source task
        #mean_demand = 2 # Target task for Scenario 1
        mean_demand = 5 # Target task for Scenario 2
        env = InventoryEnv(mean_demand=mean_demand)
    
    else:
        raise Exception('Environment {} currently not supported. Please choose among currently supported environments ["gridworld","scm"]'.format(args.env)) 
    #env = highway() # Highway Environment

    tune_hyperparams = True
    max_scores = {}
    #eval_target_policy()
    
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    transfer_modes = ['delta']
    
    if not target_task:
        # Train source task
        _, _, result_data = run()
        save_results(target_task,result_data)
    
    elif target_task:
        if not tune_hyperparams:
            _, _, result_data = run(target_task = target_task, with_transfer = with_transfer)
            save_results(target_task,result_data)
        else: 
            lr_range = [0.5,0.1,0.05,0.01]
            if with_transfer:
              for transfer_mode in transfer_modes:
                  max_scores[transfer_mode] = tune(with_transfer,lr_range, transfer_mode)    
            else:
                tune(with_transfer, lr_range)    
    print(max_scores)
