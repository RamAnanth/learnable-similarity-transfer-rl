"""
Adapted from code provided by Gimelfarb et al 2020 b: Epsilon-BMC: A Bayesian ensemble approach to epsilon-greedy exploration in model-free reinforcement learning in UAI 2019
Implements Supply Chain management environment
"""

import math
import numpy as np

class InventoryEnv(object):
    #Environment corresponding to Supply Chain management task
    def __init__(self, mean_demand = 5):
        self.PRICE = 0.5
        self.PRODUCTION_COST = 0.1
        self.STORAGE_COST = 0.02
        self.PENALTY_COST = 0.1
        self.TRUCK_COST = 0.1
        self.TRUCK_CAPACITY = 5
        self.MEAN_DEMAND_POISSON = mean_demand
        self.PRODUCTION_LIMIT = 10
        self.SHIPMENT_LIMIT = 10
        self.STORAGE_LIMIT = 50        
        self.num_actions = self.PRODUCTION_LIMIT * self.SHIPMENT_LIMIT
        self.num_states = (self.STORAGE_LIMIT + 1) ** 2
        self.index_tuple = {}
        self.index2tuple()
        self.MAX_STEPS = 200
        self.steps = 0 
        self.start = 10*(self.STORAGE_LIMIT+1)+0 #(10,0)
        
        self.reset()
        
    def index2tuple(self):
        # Convert state index to tuple for further processing
        limit = self.STORAGE_LIMIT + 1
        for x in range(self.num_states):
            self.index_tuple[x] = (x//limit,x%limit)
            
    def reset(self):
        self.steps = 0
        self.state = self.start
        return self.state

    def step(self, action):
        state = self.index_tuple[self.state]
        done = False
        info = {}
        self.steps += 1
        # what is the demand d_k?
        demand = np.random.poisson(lam=self.MEAN_DEMAND_POISSON, size=(1))
        if demand[0] > state[1]:
            penalty_cost = 0.0  # (demand[0] - state[1]) * Inventory.PENALTY_COST
            demand[0] = state[1]
        else:
            penalty_cost = 0.0
            
        # how much to ship and how much to produce
        to_ship = action // self.PRODUCTION_LIMIT
        to_produce = action - to_ship * self.PRODUCTION_LIMIT
        if to_ship > state[0]: 
            to_ship = state[0]
        if state[1] - demand[0] + to_ship > self.STORAGE_LIMIT:
            to_ship = self.STORAGE_LIMIT - state[1] + demand[0]
        if state[0] - to_ship + to_produce > self.STORAGE_LIMIT:
            to_produce = self.STORAGE_LIMIT - state[0] + to_ship
            
        # rewards and costs
        state_array = np.asarray(state)
        revenue = self.PRICE * np.sum(demand)
        production_cost = self.PRODUCTION_COST * to_produce
        storage_cost = self.STORAGE_COST * np.sum(state_array)
        penalty_cost = -self.PENALTY_COST * np.sum(((state_array < 0) * state_array)[1:])
        transport_cost = self.TRUCK_COST * math.ceil(to_ship / self.TRUCK_CAPACITY)
        net_reward = revenue - production_cost - storage_cost - transport_cost - penalty_cost
            
        # state update
        new_factory_inventory = state[0] - to_ship + to_produce
        new_store_inventory = state[1] - demand[0] + to_ship
        new_state = (new_factory_inventory, new_store_inventory)
        
        self.state = new_state[0]*(self.STORAGE_LIMIT+1) + new_state[1]
        
        if self.steps == self.MAX_STEPS:
            done = True
            info = {"termination_condition":"Maximum number of steps limit reached"}
            return self.state, net_reward,done,info
        
        return self.state, net_reward, done, info
    
    """
    def default_encoding(self, state):
        arr = np.zeros(self.num_states, dtype=np.float32)
        arr[state[0]] = 1.0
        arr[self.STORAGE_LIMIT + 1 + state[1]] = 1.0
        arr = arr.reshape((1, -1))
        return arr
    """