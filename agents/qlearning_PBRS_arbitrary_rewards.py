import numpy as np
import random
"""
Author : Ram and Hyun-Rok
"""

class Qlearning_transfer_PBRS_arbitrary_reward(object):
    # Simple q-learning + static advice through explicit shaping

    def __init__(self, env, alpha, beta, alpha_2, source_policies, source_values, delta = None, similarity = 'delta'):
        self.env = env
        self.l_rate = alpha
        self.count = 0
        self.l_rate_sec_q = alpha_2 
        self.l_rate_delta = beta
        self.gamma = 0.99
        self.source_policies = source_policies
        self.source_values = source_values
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.similarity = similarity 
        
        # Decay parameters
        #self.kappa = 1
        #self.kappa_decay = 1/(200*2000)
    
        self.q_table = np.zeros((self.num_states, self.num_actions))    
        self.delta_sa = np.zeros((self.num_states,self.num_actions))
        self.delta_s = np.zeros((self.num_states))
        
        print("Q Learning with Potential function as ", self.similarity)
        
        """
        # Initialize Q table with state values
        if self.transfer_mode == 'delta':
            for state in range(self.num_states):
                self.q_table[state:,] = np.max(self.source_values[0][state])
        """            
        self.phi = np.zeros((self.num_states,self.num_actions))
        
    def train(self, transition_tuple, iter):

        s, a, r, s1, done = transition_tuple # (s,a,r,s',terminate_condition)
        # s, a, r, s1, a1, done = transition_tuple  # (s,a,r,s',terminate_condition)
        
        
        # Fixed Learning Rate
        current_l_rate = self.l_rate
        current_l_rate_sec = self.l_rate_sec_q 
        current_l_rate_delta = self.l_rate_delta
        
        """
        # Decaying learning rate
        decay_rate = 0.999
        current_l_rate = self.l_rate * np.power(decay_rate, iter)
        current_l_rate_sec = self.l_rate_sec_q * np.power(decay_rate, iter)
        current_l_rate_delta = self.l_rate_delta * np.power(decay_rate, iter)
        """
        
        # For arbitrary reshaped reward
        q_s = self.source_values[s,a]
        adv_s = self.source_values[s,a] - np.max(self.source_values[s]) 
        r_phi = - q_s
        
        if self.similarity == 'delta':
            #r_phi = self.delta_sa[s,a]
            r_phi = self.delta_s[s]
            #self.delta_sa[s, a] = (1 - current_l_rate_delta) * self.delta_sa[s, a] + current_l_rate_delta * (r + self.gamma * np.max(self.source_values[s1]) - self.source_values[s, a])   
            self.delta_s[s] = (1 - current_l_rate_delta) * self.delta_s[s] + current_l_rate_delta * (r + self.gamma * np.max(self.source_values[s1]) - self.source_values[s,a])
            

        # Compute reshaped reward

        a1 = np.argmax(self.q_table[s1])        
        F = self.gamma * self.phi[s1,a1] - self.phi[s,a]

        if done:
            phi_target = r_phi - self.phi[s,a]
        else:
            phi_target = r_phi + F
        
        self.phi[s,a] = self.phi[s,a] + current_l_rate_sec * (phi_target)
        
        if done:
            q_target = r - self.phi[s,a]
        else:
            q_target = r + self.phi[s1,a1] - self.phi[s,a] + self.gamma * np.max(self.q_table[s1])
            
        self.q_table[s, a] = (1 - current_l_rate) * self.q_table[s, a] + current_l_rate * q_target
    
    def bias(self,state):
        # Retrieve the source advantage function to be used as advice for a particular state 
        source_value_s = np.max(self.source_values[state])
        adv_s = self.source_values[state] - source_value_s
        return adv_s
        
    def eps_greedy_action(self, state, epsilon):
        s = state
        
        if random.random() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[s])    
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

    def getPolicy(self):
        return np.argmax(self.q_table, axis=1)
    
    def getDelta(self):
        return self.delta_s[0]
