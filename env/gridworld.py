'''
@author: Ram

Implements Windy Gridworld Environment

'''

import numpy as np 
import random
import matplotlib.pyplot as plt 


np.random.seed(0)
random.seed(0)

class GridWorldEnv(object):
    # Environment corresponding to the Windy Gridworld task
    
    def __init__(self, num_rows = 7, num_cols = 10  , wind = None):    
        self.shape = num_rows,num_cols
        self.wind = wind # Determines wind direction and magnitude for each column
        self.index_grid = {}
        
        # Map actions to numbers
        self.LEFT = 0
        self.UP = 1
        self.RIGHT = 2
        self.DOWN = 3
        
        
        self.num_states = num_rows*num_cols
        self.num_actions = 4
        self.MAX_STEPS = 200
        self.index2grid()
        self.steps = 0
        #self.start = (3,0)
        #self.goal = (17,14)
        
        # Set start and goal states
        self.start = 3*20 + 0 # (3,0)
        self.goal = 17*20 + 14 #(17,14)
        self.reset()
        
    def index2grid(self):
        # Convert state index to grid coordinates
        for x in range(self.shape[0]*self.shape[1]):
            self.index_grid[x] = (x//self.shape[1],x%self.shape[1])
    
    def reset(self):
        self.steps = 0
        self.state = self.start
        return self.state
    
    def step(self, action):
        row, col = self.index_grid[self.state] 
        #row, col = self.state
        done = False
        rew = -1
        info = {}
        self.steps += 1
        
        if self.state==self.goal:
            done = True
            info = {"termination_condition":"Reached Goal"}
            return self.state,rew,done,info
        
        if self.steps == self.MAX_STEPS:
            done = True
            info = {"termination_condition":"Maximum number of steps limit reached"}
            return self.state,rew,done,info
            """
		With probability 0.25 end up taking random action and probability 0.75 take intended action
        
		Alternate way to implement stochasticity in action where 25% of time you move in random direction
		"""
        if random.random() < 0.25:
            action = np.random.randint(self.num_actions)
        
        if action == self.LEFT:
            diff = np.array([-self.wind[col],-1])
        
        elif action == self.UP:
            diff = np.array([-1-self.wind[col],0])
            
        elif action == self.DOWN:
            diff = np.array([1-self.wind[col],0])
        
        elif action == self.RIGHT:
            diff = np.array([-self.wind[col],1])
        
        else:
            raise Exception('Invalid Action {}. Action should be 0,1,2 or 3'.format(action))
        
        row_x, col_y = np.clip(np.array(self.index_grid[self.state])+diff,[0,0],[self.shape[0]-1,self.shape[1]-1])
        self.state = row_x*self.shape[1]+col_y
        #row_x, col_y = np.clip(np.array(self.state) + diff, [0,0], [self.shape[0]- 1, self.shape[1] - 1]) 
        #self.state = row_x, col_y
        
        return self.state,rew,done,info
					
	# def visualise_grid(self):
		
	# 	grid = np.zeros((self.grid_size,self.grid_size))


	# 	plt.imshow(~self.grid, cmap='gray', interpolation='nearest')
	# 	plt.xticks([]), plt.yticks([])
	# 	plt.show()


def heuristic_solver():
	# Optimal policy obtained heuristically
	action_seq = np.zeros(15)
	action_seq[:9] = 2
	action_seq[9:13] = 3
	return action_seq

if __name__ == '__main__':
	
  # Test environment using parameters similar to the one in Sutton and Barto
	wind = np.array([0,0,0,1,1,1,2,2,1,0])

	env = GridWorldEnv(7,10,wind=wind)
	max_steps = 50

	obs = env.reset()

	optimal_seq = heuristic_solver()
	episode_ret = 0

	for i in range(15):
		action = optimal_seq[i] 
		obs,reward,done,_ = env.step(action)
		episode_ret+=reward
		if done:
			break

	print("Final state",obs)
	print("Episode Return",episode_ret)

# grid.visualise_grid()



