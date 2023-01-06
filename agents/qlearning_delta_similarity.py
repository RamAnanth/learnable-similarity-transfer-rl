# Author : Ram 
import numpy as np
import random 
from collections import defaultdict

np.random.seed(0)
random.seed(0)

class Qlearning_Delta(object):
    # Class for implementing the algorithm with delta as a criterion for direct transfer
    def __init__(self, env, alpha, beta, pi_source, q_source):
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        self.gamma = 0.99 # Discount Factor
        self.alpha = alpha # Learning rate for Q value
        self.beta = beta # Learning rate for delta
        self.threshold_start = 5 # Threshold value
        
        self.pi_source = pi_source
        self.q_source = q_source
        
        # Possibly decay the threshold to converge to target optimal
        self.threshold = self.threshold_start
        self.threshold_end = 0
        self.threshold_decay = 10000
        
        self.advantage = self.q_source - np.max(self.q_source, axis = 1, keepdims = True)
        
        self.max_advantage = np.max(np.abs(self.q_source))
        
        print(self.max_advantage)
        
        self.source_count = 0
        self.total_count = 0
        self.episodes_done = 0
        
        self.Q = np.zeros((self.num_states,self.num_actions))
        self.delta_sa = np.zeros((self.num_states,self.num_actions))
        
        """
        #Initialize with Source value function
        for state in range(self.num_states):
                self.delta_sa[state:,] = np.max(self.q_source[state])
        """
        print('Q Learning with Delta as Criterion for direct Transfer')
    
    def train(self,transition_tuple, iter):
        s,a,r,s1,done= transition_tuple
        
        self.delta_sa[s, a] = (1 - self.beta) * self.delta_sa[s, a] + self.beta * (r + self.gamma * np.max(self.q_source[s1]) - self.q_source[s, a])
        
        if done:
            Q_target = r
            self.episodes_done = self.episodes_done + 1
            self.threshold = max(0, self.threshold)
            self.count_percentage = self.source_count/self.total_count
            self.source_count = 0
            self.total_count = 0
            #self.threshold = self.threshold_end + (self.threshold_start - self.threshold_end) * np.exp(-1. * self.episodes_done / self.threshold_decay) #Decaying the threshold  
        else:
            Q_target = r + self.gamma * np.max(self.Q[s1])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * Q_target

    def eps_greedy_action(self,state, epsilon):
        s = state
        # Find max delta value
        if np.max(np.abs(self.delta_sa)) <= self.threshold:
            self.source_count = self.source_count + 1
            action = self.pi_source[s]
        else:
			      if random.random()<epsilon:
				        action = np.random.randint(self.num_actions)
			      else:
				        action = np.argmax(self.Q[s])
        self.total_count = self.total_count + 1
        return action
  
    def getQvalues(self):
        return self.Q # q-value
        # return np.max(self.q_table, axis=1)  # state-value
    
    def getPolicy(self):
        return np.argmax(self.Q, axis=1)
    
    def getDelta(self):
        return np.max(np.abs(self.delta_sa), axis = 1)
    
    def getCountPercentage(self):
        return self.count_percentage
    