import numpy as np
import random
from collections import defaultdict
"""
Author : Ram and Hyun-Rok
"""

np.random.seed(0)
random.seed(0)

class Qlearning():
    # Simple q-learning

    def __init__(self, env, gamma, alpha):
        self.env = env
        self.l_rate = alpha # Learning rate for Q value
        self.discount_rate = gamma #Discount Factor

        self.num_states = env.num_states
        self.num_actions = env.num_actions

        self.q_table = np.zeros((self.num_states, self.num_actions))
        print("Q Learning")

        
    def train(self, transition_tuple, iter):

        s, a, r, s1, done = transition_tuple # (s,a,r,s',terminate_condition)

        current_l_rate = self.l_rate
        
        if done:
            q_target = r
        else:
            q_target = r + self.discount_rate * np.max(self.q_table[s1])

        self.q_table[s, a] = (1 - current_l_rate) * self.q_table[s, a] + current_l_rate * q_target
    

    def eps_greedy_action(self, state, epsilon):
        s = state

        if random.random() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[s])

        return action

    def getQvalues(self):
        #return self.Q
        return self.q_table # q-value
        # return np.max(self.q_table, axis=1)  # state-value

    def getPolicy(self):
        return np.argmax(self.q_table, axis=1)
