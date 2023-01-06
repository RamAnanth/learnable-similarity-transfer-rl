# Author : Ram 
import numpy as np
import random 
from scipy.special import softmax

np.random.seed(0)
random.seed(0)

class Qlearning_Multi_Delta(object):
    # Class for implementing the algorithm with delta as a criterion for direct transfer
    def __init__(self, env, alpha, beta, pi_sources, q_sources):
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        self.gamma = 0.99 # Discount Factor
        self.alpha = alpha # Learning rate for Q value
        self.beta = beta # Learning rate for delta
        self.threshold_start = 1 # Threshold value
        
        # Possibly decay the threshold to converge to target optimal
        self.threshold = self.threshold_start
        self.threshold_end = 0
        self.threshold_decay = 4000
        self.source_count = 0
        self.total_count = 0
        self.episodes_done = 0
        self.num_source = len(q_sources)
        self.counts = np.zeros(self.num_source)
        self.Q = np.zeros((self.num_states,self.num_actions))
        self.delta_sa = np.zeros((self.num_source, self.num_states,self.num_actions))
        self.pi_sources = pi_sources
        self.q_sources = q_sources
        
        """
        #Initialize with Source value function
        for state in range(self.num_states):
                self.delta_sa[state:,] = np.max(self.q_source[state])
        """
        print('Q Learning with Multiple sources using Delta as Criterion for direct Transfer')
    
    def train(self,transition_tuple, iter):
        s,a,r,s1,done= transition_tuple

        for i in range(self.num_source):
            self.delta_sa[i][s, a] = (1 - self.beta) * self.delta_sa[i][s, a] + self.beta * (r + self.gamma * np.max(self.q_sources[i][s1]) - self.q_sources[i][s, a])
        
        if done:
            Q_target = r
            self.episodes_done = self.episodes_done + 1
            self.source_count = 0
            self.total_count = 0
            self.count_percentage = self.counts
            self.counts = np.zeros(self.num_source)
            self.threshold = self.threshold_end + (self.threshold_start - self.threshold_end) * np.exp(-1. * self.episodes_done / self.threshold_decay) #: Decaying the threshold  
        else:
            Q_target = r + self.gamma * np.max(self.Q[s1])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * Q_target

    def eps_greedy_action(self,state, epsilon):
        s = state
        if random.random()<epsilon:
            action = np.random.randint(self.num_actions)
        else:
            if random.random()<=self.threshold:
                max_deltas = [-np.max(np.abs(self.delta_sa[i])) for i in range(self.num_source)]
                index = np.random.choice(np.arange(self.num_source), p = softmax(max_deltas))
                action = self.pi_sources[index][s]
                self.counts[index] = self.counts[index] + 1
            else:
                action = np.argmax(self.Q[s])
        
        return action
  
    def getQvalues(self):
        return self.Q # q-value
        # return np.max(self.q_table, axis=1)  # state-value
    
    def getPolicy(self):
        return np.argmax(self.Q, axis=1)
    
    #def getDelta(self):
    #    return np.max(np.abs(self.delta_sa), axis = 1)
    
    def getCountPercentage(self):
        return self.count_percentage
    