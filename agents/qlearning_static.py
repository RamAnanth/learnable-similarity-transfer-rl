import numpy as np
import random

"""
Author : Ram and Hyun-Rok
"""

class Qlearning_transfer_PBRS_static(object):
    # Simple q-learning + testing of different static potential functions

    def __init__(self, env, alpha, beta, source_policies, source_values, delta = None, transfer_mode = 'delta'):
        self.env = env
        self.l_rate = alpha
        self.l_rate_delta = beta
        self.discount_rate = 0.99
        self.source_policies = source_policies
        self.source_values = source_values
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.transfer_mode = transfer_mode 
        self.num_sources = len(source_policies)
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        if delta is None:
            self.delta_sa = np.zeros((self.num_sources, self.num_states,self.num_actions))
            self.delta_s = np.zeros((self.num_sources, self.num_states))
        else:
            self.delta_i = delta.reshape(1,len(delta))
        #self.delta_sa = np.zeros((self.num_sources, self.num_states, self.num_actions))
        
        print("Q Learning with Static Biased PBRS using ", self.transfer_mode)
        
        """
 
        # Initialize Q table with state values
        if self.transfer_mode == 'delta':
            for state in range(self.num_states):
                self.q_table[state:,] = np.max(self.source_values[0][state])
        """
                      
        #if self.transfer_mode in ['policy','delta_action','advantage','qvalue', 'random']:
        self.phi = np.zeros((self.num_states,self.num_actions))
        #else:
        #    self.phi = np.zeros(self.num_states)
        

    def train(self, transition_tuple, iter):

        s, a, r, s1, done = transition_tuple # (s,a,r,s',terminate_condition)
        # s, a, r, s1, a1, done = transition_tuple  # (s,a,r,s',terminate_condition)
        
        # Fixed Learning Rate
        current_l_rate = self.l_rate 
        current_l_rate_delta = self.l_rate_delta
        
        """
        # Decaying learning rate
        decay_rate = 0.999
        current_l_rate = self.l_rate * np.power(decay_rate, iter)
        current_l_rate_delta = self.l_rate_delta * np.power(decay_rate, iter)
        """
        
        # Different Sources of potential function
        source_value_s = np.max(self.source_values[0][s])
        source_value_next_s = np.max(self.source_values[0][s1])
        adv_s = self.source_values[0][s,a] - source_value_s
        adv_next_s = self.source_values[0][s1, np.argmax(self.q_table[s1])] - source_value_next_s
        a1 = np.argmax(self.q_table[s1])
        
        #current_delta = self.delta_s[0,s]
        if self.transfer_mode == 'delta_action':
            current_delta = self.delta_sa[0, s,a]
            next_delta = self.delta_sa[0,s1,a1]
        elif self.transfer_mode == 'delta':
            current_delta = self.delta_s[0, s]
            next_delta = self.delta_s[0, s1]
        
        for i in range(self.num_sources):
            if self.transfer_mode == 'delta':
                self.delta_s[i, s] = (1 - current_l_rate_delta) * self.delta_s[i, s] + current_l_rate_delta * (r + self.discount_rate * np.max(self.source_values[i][s1]) - self.source_values[i][s, a])
            elif self.transfer_mode == 'delta_action':
                self.delta_sa[i, s, a] = (1 - current_l_rate_delta) * self.delta_sa[i, s, a] + current_l_rate_delta * (r + self.discount_rate * np.max(self.source_values[i][s1]) - self.source_values[i][s, a])
        
                
        # For arbitrary reshaped reward
        if self.transfer_mode == 'qvalue':
            self.phi[s,a] =  self.source_values[0][s,a]
            self.phi[s1,a1] = self.source_values[0][s1,a1]
        elif self.transfer_mode == 'policy':
            self.phi[s,a] = 1 if a == self.source_policies[0][s] else 0
            self.phi[s1,a1] = 1 if a1 == self.source_policies[0][s1] else 0  
        elif self.transfer_mode in ['delta','delta_action']:
            self.phi[s,a] = - current_delta
            self.phi[s1,a1] = - next_delta
        elif self.transfer_mode =='value':
            self.phi[s] = source_value_s
            self.phi[s1] = source_value_next_s
        elif self.transfer_mode =='advantage':
            self.phi[s,a] = adv_s
            self.phi[s1,a1] = adv_next_s
        elif self.transfer_mode == 'random':
            self.phi[s1,a1] = np.random.uniform(0,5)
            self.phi[s,a] = np.random.uniform(0,5)
        # Compute reshaped reward

        
        #F = self.discount_rate * self.phi[s1] - self.phi[s]
        #if self.transfer_mode in ['policy','delta_action','advantage','qvalue', 'random']:
        F = self.discount_rate * self.phi[s1,a1] - self.phi[s,a]
        if self.transfer_mode =='policy':
            scale = 2
            F = F * scale
        self.F = F
        #else:
        #    F = self.discount_rate * self.phi[s1] - self.phi[s]
        
        if done:
            #if self.transfer_mode in ['value','delta']:
            #    phi_target = -r_phi - self.phi[s]
            #    q_target = r - self.phi[s]
            #else:
            q_target = r - self.phi[s,a]
        else:
            q_target = r + F + self.discount_rate * np.max(self.q_table[s1])
            
        self.q_table[s, a] = (1 - current_l_rate) * self.q_table[s, a] + current_l_rate * q_target

    def eps_greedy_action(self, state, epsilon):
        s = state
        if random.random() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[s] + self.phi[s])
        return action

    def eps_greedy_action_test(self, state, epsilon):
        s = self.state_index[tuple(state)]

        if random.random() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[s])
        return action

    def getQvalues(self):
        return self.q_table # q-value
        # return np.max(self.q_table, axis=1)  # state-value
    
    def reshaped_reward(self):
        return self.F

    def getPolicy(self):
        return np.argmax(self.q_table, axis=1)
    
    def getDelta(self):
        return self.delta_sa[0]