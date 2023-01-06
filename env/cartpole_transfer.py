"""
Extend OpenAI gym CartPole-v0 implementation to include different gravities to design transfer learning scenario
"""

from gym.envs.classic_control import cartpole

class CartPoleEnv_transfer(cartpole.CartPoleEnv):

    def __init__(self,dynamics_factor = 0.5):
        super(CartPoleEnv_transfer, self).__init__()
        self.max_episode_steps = 200 # v0
        self._elapsed_steps = None
        self.name = 'CartPole'
        #self.length = 0.5 * dynamics_factor
        #self.masspole = 0.1*dynamics_factor
        self.gravity = 9.8 * dynamics_factor
        #self.force_mag = 1000.0 * dynamics_factor
        #self.polemass_length = self.polemass_length * dynamics_factor

    def step(self, action):
        observation, reward, done, info = super(CartPoleEnv_transfer, self).step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return super(CartPoleEnv_transfer, self).reset()