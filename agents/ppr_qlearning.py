# Author : Ram 
# Based on Fernando and Veloso 2006 AAMAS

import numpy as np
import random 

class PPR_Qlearning(object):
    # Q learning with pi-reuse exploration strategy
    def __init__(self, env, alpha, pi_source= None):
		#NOTE:
		# The parameter v in the paper that decays beta takes same value as gamma
		# as given in some of the examples
		
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.gamma = 0.99
        self.alpha = alpha
        self.psi = 1.0
        self.count = 0
        self.Q = np.zeros((self.num_states,self.num_actions))
        self.pi_source = pi_source
        print('Q Learning with pi-reuse action selection')

    def train(self,transition_tuple, iter):
        s,a,r,s1,done= transition_tuple
        if done:
            self.psi = self.psi*0 # Decay psi every episode
            Q_target = r
        else:
			      Q_target = r + self.gamma * np.max(self.Q[s1])
        
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * Q_target
    
    def eps_greedy_action(self,state, epsilon): 

		    s = state 

		    if random.random() <= self.psi:
			      self.count = self.count + 1
			      action = self.pi_source[s]
		    else:
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