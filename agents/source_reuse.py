import numpy as np
import random
"""
Author : Ram 
"""

np.random.seed(0)
random.seed(0)

class Source():
    # Simple testing of how source policies perform directly in the target and learning delta function simultaneously

    def __init__(self, env, q_source):
        self.env = env
        print("Deploying source optimal policy directly")
        
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.gamma = 0.99

        self.q_table = np.zeros((self.num_states, self.num_actions))
        #self.delta_sa = np.zeros((self.num_states,self.num_actions))
        self.q_source = q_source
        self.q_table = q_source
        
    def train(self, transition_tuple, iter):
        s,a,r,s1,done= transition_tuple
        #self.delta_sa[s, a] = (1 - self.beta) * self.delta_sa[s, a] + self.beta * (r + self.gamma * np.max(self.q_source[s1]) - self.q_source[s, a])
        
    def eps_greedy_action(self, state, epsilon):
        s = state
        action = np.argmax(self.q_table[s])
        return action

    def getQvalues(self):
        #return self.Q
        return self.q_table # q-value
        # return np.max(self.q_table, axis=1)  # state-value

    def getPolicy(self):
        return np.argmax(self.q_table, axis=1)
        #return {k: np.argmax(v) for k, v in self.Q.items()}
    
    def getDelta(self):
        return np.max(np.abs(self.delta_sa))