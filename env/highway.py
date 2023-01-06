import numpy as np
import random

from highway_env.envs.highway_env import HighwayEnv

class highway_env(HighwayEnv):
    def __init__(self,vehicles_count = 50):
        super(highway_env, self).__init__()

if __name__ == '__main__':
    env = highway_env()
    print(env.reset())
    