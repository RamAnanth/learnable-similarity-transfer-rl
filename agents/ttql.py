# Author : Ram 
import numpy as np
import random 

np.random.seed(0)
random.seed(0)

class TTQlearning(object):
    "Implementation attempt for Target transfer Q Learning algorithm from Wang et al 2020, Neurocomputing"
    
    def __init__(self, env, alpha, q_source):
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.gamma = 0.99
        self.alpha = alpha
    
        self.episodes_done = 0
        self.Q = np.zeros((self.num_states,self.num_actions))
        self.q_source = q_source
        
        print('Target Transfer Q Learning')
    
    def MNBE(self, transition_tuple,Q):
        "Function to compute maximum bellman error"
        s,a,r,s1,done= transition_tuple
        if done:
            mnbe_target = r
        else:
            mnbe_target = r + self.gamma*np.max(Q[s1])
        mnbe = np.max(np.abs(Q-mnbe_target))
        return mnbe
       
    def error_condition(self, transition_tuple):
        return self.MNBE(transition_tuple,self.q_source) <= self.MNBE(transition_tuple,self.Q)
        
    def train(self,transition_tuple, iter):
        s,a,r,s1,done= transition_tuple
        if done:
            Q_target = r
            self.episodes_done = self.episodes_done + 1            
        else:
            if self.error_condition(transition_tuple):
                Q_target = r + self.gamma * np.max(self.q_source[s1])
            else:
                Q_target = r + self.gamma * np.max(self.Q[s1])
                
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * Q_target
        
    def eps_greedy_action(self,state, epsilon):
        s = state
        if random.random()<epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[s])
        return action
  
    def getQvalues(self):
        return self.Q # q-value
        # return np.max(self.q_table, axis=1)  # state-value
    
    def getPolicy(self):
        return np.argmax(self.Q, axis=1)
    