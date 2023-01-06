"""
Extend OpenAI gym Acrobot-v1 implementation to include different link masses to design transfer learning scenario
"""

from gym.envs.classic_control import acrobot

class AcrobotEnv_transfer(acrobot.AcrobotEnv):

    def __init__(self, moi = 1):
        super(AcrobotEnv_transfer, self).__init__()
        #self.LINK_MASS_1 = 1.0*link_mass
        #self.LINK_MASS_2 = 1.0*link_mass
        self.LINK_MOI = moi
        self.name = 'Acrobot'
        self.max_episode_steps = 500 # Set upper limit
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, info = super(AcrobotEnv_transfer, self).step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return super(AcrobotEnv_transfer, self).reset()