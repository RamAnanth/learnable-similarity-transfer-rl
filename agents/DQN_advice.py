# DQN with Shaping based on different sources of Advice: Code for DQN adapted from PyTorch tutorial on DQN
import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action',  'reward', 'next_state', 'next_action'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(20)

class MlpPolicy(nn.Module):
    # Network that is used to define quantities like Q, delta
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
        
class DQN_Advice(object):
    # Class that defines the algorithm for DQN with different forms of advice
    def __init__(self,env,source_path,lr = 1e-4,replay_capacity = 10000, transfer_mode = 'delta'):
        
        print('Experiment to compare different choices of advice')
        # Neural Network architecture
        if env.name == 'CartPole':
            num_neurons = 64
        elif env.name in ['Acrobot', 'LunarLander', 'Reacher']:
            num_neurons = 256
        hidden_layers = [num_neurons,num_neurons] # Cartpole : 64 64
        state_dim = env.observation_space.shape[0]
        
        
        self.action_dim = env.action_space.n
        self.policy_net = MlpPolicy(state_dim,self.action_dim,hidden_layers)
        self.target_net = MlpPolicy(state_dim,self.action_dim,hidden_layers)
        
        self.env = env
        self.transfer_mode = transfer_mode
        self.similarity = None
        if self.transfer_mode == 'static_delta':
            self.transfer_mode = 'static'
            self.similarity = 'delta'
        
        if self.transfer_mode == 'delta_similarity':
            self.transfer_mode = 'advantage'
            self.similarity = 'delta'
            
        print("Data",(self.transfer_mode,self.similarity))
                      
        self.source_net = MlpPolicy(state_dim,self.action_dim,hidden_layers)
        
        # Delta as a function of state and action
        self.delta =  MlpPolicy(state_dim,self.action_dim,hidden_layers) # Cartpole : 64 64
        if self.transfer_mode =='delta_action':
            self.delta =  MlpPolicy(state_dim,self.action_dim,hidden_layers)
        
        if self.transfer_mode in ['policy','delta_action','advantage','qvalue']:
            self.phi =  MlpPolicy(state_dim,self.action_dim,hidden_layers) # Cartpole : 64 64
            self.target_phi = MlpPolicy(state_dim,self.action_dim,hidden_layers)
        else:
            self.phi =  MlpPolicy(state_dim,1,hidden_layers) # Cartpole : 64 64
            self.target_phi = MlpPolicy(state_dim,1,hidden_layers)
        
        # Load the saved source network
        self.source_net.load_state_dict(torch.load(source_path))
        self.grad_norm = 10
        
        # Set the decay parameters
        self.kappa = 1
        self.kappa_decay = 5e-6
        
        if self.transfer_mode == 'no_transfer':
            self.kappa = 0
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.replay_buffer = ReplayMemory(replay_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_phi = optim.Adam(self.phi.parameters(),lr=5*lr)       
        self.optimizer_delta = optim.Adam(self.delta.parameters(),lr=5*lr )
        
        # Exploration parameters 
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 500
        self.steps_done = 0
        
        self.gamma = 0.99 # Discount factor
        self.scale = 1
    
    def learn_phi(self, transition_tuple):
        "Function for updating the dynamic potential function based on the chosen quantity from the source task, if used as advice"
        
        state,action,reward,next_state,next_action = transition_tuple
        batch = Transition(state,action,reward,next_state,next_action)
        batch_size = 1
  
        non_final_mask = False if batch.next_state is None else True
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        
        non_final_next_state = batch.next_state if batch.next_state is not None else None
        non_final_next_action = batch.next_action if batch.next_action is not None else None
     
        # Compite current value of delta from the neural network and calculate its target
        current_delta = self.delta(state_batch).gather(1,action_batch)
        source_action_values = self.source_net(state_batch).gather(1, action_batch).detach()
        source_next_state_values = torch.zeros(batch_size, device=device)
        if non_final_mask:
            source_next_state_values = self.source_net(non_final_next_state).max(1)[0].detach()
        expected_delta = (source_next_state_values * self.gamma) + reward_batch - source_action_values.squeeze(1)

        if self.transfer_mode=='delta_action':
            source_action_values = self.source_net(state_batch).gather(1, action_batch).detach()
            state_phi = self.phi(state_batch).gather(1,action_batch).squeeze(1)
            current_delta = self.delta(state_batch).gather(1,action_batch)
            
            next_state_phi = torch.zeros(batch_size, device=device)
            #next_state_phi[non_final_mask] = self.phi(non_final_next_states).gather(1,non_final_next_actions).squeeze(1).detach()
            #next_state_phi[non_final_mask] = self.target_phi(non_final_next_states).gather(1,non_final_next_actions).squeeze(1).detach()
            
            source_next_state_values = torch.zeros(batch_size, device=device)
            #source_next_state_values[non_final_mask] = self.source_net(non_final_next_states).max(1)[0].detach()
            
            if non_final_mask:
                next_state_phi = self.target_phi(non_final_next_state).gather(1,non_final_next_action).squeeze(1).detach()
                source_next_state_values = self.source_net(non_final_next_state).max(1)[0].detach()
                
            expected_delta = (source_next_state_values * self.gamma) + reward_batch - source_action_values.squeeze(1)
            
            shaping_term = (next_state_phi*self.gamma) - state_phi
            
            reshaping_term = shaping_term.detach()    
        
        elif self.transfer_mode == 'policy':
            source_actions = self.source_net(state_batch).max(1)[1].detach()
            target_actions = self.policy_net(state_batch).max(1)[1].detach()            
            state_phi = self.phi(state_batch).gather(1,action_batch).squeeze(1)
                      
            next_state_phi = torch.zeros(batch_size, device=device)
            #next_state_phi[non_final_mask] = self.phi(non_final_next_states).gather(1,non_final_next_actions).squeeze(1).detach()
            
            state_phi = self.phi(state_batch).gather(1,action_batch).squeeze(1)
                      
            if non_final_mask:
                next_state_phi = self.target_phi(non_final_next_state).gather(1,non_final_next_action).squeeze(1).detach()
                
            shaping_term = (next_state_phi*self.gamma) - state_phi
        
            reward_scale = 100
            expected_phi = reward_scale*(source_actions==target_actions).float()
            
            reshaping_term = shaping_term.detach()
        
            
        elif self.transfer_mode == 'advantage':
            
            # Potential value for current state, action 
            state_phi = self.phi(state_batch).gather(1,action_batch).squeeze(1)
            
            next_state_phi = torch.zeros(batch_size, device=device)
            
            if non_final_mask:
                # Using a target network compute the value of potential function for next state, next action
                next_state_phi = self.target_phi(non_final_next_state).gather(1,non_final_next_action).squeeze(1).detach()
            shaping_term = (next_state_phi*self.gamma) - state_phi
            
            source_action_values = self.source_net(state_batch).gather(1, action_batch).squeeze(1).detach()
            source_state_values = self.source_net(state_batch).max(1)[0].detach()
            advantage = source_action_values - source_state_values            
            expected_phi = advantage
 
        elif self.transfer_mode == 'qvalue':
            
            state_phi = self.phi(state_batch).gather(1,action_batch).squeeze(1)
            
            next_state_phi = torch.zeros(batch_size, device=device)
            
            #next_state_phi[non_final_mask] = self.phi(non_final_next_states).gather(1,non_final_next_actions).squeeze(1).detach()
            if non_final_mask:
                next_state_phi = self.target_phi(non_final_next_state).gather(1,non_final_next_action).squeeze(1).detach()
            #next_state_phi[non_final_mask] = self.target_phi(non_final_next_states).gather(1,non_final_next_actions).squeeze(1).detach()

            shaping_term = (next_state_phi*self.gamma) - state_phi
            
            source_action_values = self.source_net(state_batch).gather(1, action_batch).detach()
            expected_phi  =  source_action_values.squeeze(1)            
            reshaping_term = shaping_term.detach()
        
        if self.transfer_mode in ['delta','delta_action', 'static'] :
            """Update delta"""
            expected_phi = -current_delta.squeeze(1)
            loss_delta = F.smooth_l1_loss(current_delta,expected_delta.unsqueeze(1))
            #loss_phi = F.smooth_l1_loss(shaping_term,expected_phi)
            self.optimizer_delta.zero_grad()
            #self.optimizer_phi.zero_grad()
            loss_total = loss_delta
            loss_total.backward()
            #torch.nn.utils.clip_grad_norm_(self.phi.parameters(), self.grad_norm)
            #self.optimizer_phi.step()
            torch.nn.utils.clip_grad_norm_(self.delta.parameters(), self.grad_norm)
            self.optimizer_delta.step()
            
        elif self.transfer_mode in ['policy' ,'advantage','value','qvalue']:
            loss_phi = F.smooth_l1_loss(shaping_term,expected_phi)
            loss_delta = F.smooth_l1_loss(current_delta,expected_delta.unsqueeze(1))
            self.optimizer_delta.zero_grad()
            self.optimizer_phi.zero_grad()
            loss_total = loss_phi + loss_delta
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.delta.parameters(), self.grad_norm)
            torch.nn.utils.clip_grad_norm_(self.phi.parameters(), self.grad_norm)
            self.optimizer_delta.step()
            self.optimizer_phi.step()
            
            
    def static_bias(self,state):
        # Calculate Advantage function to be used as static bias
        source_action_values = self.source_net(state).detach()
        source_state_values = self.source_net(state).max().detach()
        advantage = source_action_values - source_state_values
        return -advantage            
          
    def learn(self):
        # Function to update the Q values
        
        if self.env.name == 'CartPole':
            self.batch_size = 64 # Cartpole : 64
            target_interval = 2000
            
        elif self.env.name in ['Acrobot', 'LunarLander', 'Reacher']:
            self.batch_size = 128 # Acrobot, LunarLander : 64
            target_interval = 1000
        
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))
        
        # Target network update
        if self.steps_done % target_interval ==0 :
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_phi.load_state_dict(self.phi.state_dict())
            
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        non_final_next_actions = torch.cat([a for a in batch.next_action
                                                if a is not None]) 
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        
    # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_norm)
        self.optimizer.step()
        
    def select_action(self,state):
        # Select the action to be used by the environment
        
        # Based on dynamic advice/static advice
        if self.transfer_mode!= 'static':
            bias = self.phi(state).detach()
        else:
            bias = self.static_bias(state).detach()
            
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done  = self.steps_done + 1
        self.kappa = max(0,self.kappa - self.kappa_decay)
           
        if sample > eps_threshold:
            with torch.no_grad():
                # Compute the bias factor based on the chosen algorithm
                
                bias_decay = self.kappa

                return (self.policy_net(state) - bias_decay*bias).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device)
                
    def select_greedy_action(self,state):
        # Select the biased greedy action
        
        if self.transfer_mode!= 'static':
            bias = self.phi(state).detach()
        else:
            bias = self.static_bias(state).detach()
        with torch.no_grad():
            bias_decay = self.kappa
            return (self.policy_net(state) - bias_decay*bias).max(0)[1].view(1, 1)

    def select_eval_action(self,state):
        # Select the greedy action for purpose of evaluation
        with torch.no_grad():
            return (self.policy_net(state)).max(0)[1].view(1, 1)
            
class ReplayMemory(object):
    # Replay Buffer
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
