from gym.envs.classic_control import cartpole

class CartPoleEnv_source(cartpole.CartPoleEnv):

    def __init__(self,gravity_factor = 0.5):
        super(CartPoleEnv_source, self).__init__()
        self.max_episode_steps = 200 # v3
        self._elapsed_steps = None
        self.gravity = 9.8*gravity_factor
        # self.force_mag = 1000.0
        # self.polemass_length = self.polemass_length * 0.5

    def step(self, action):
        observation, reward, done, info = super(CartPoleEnv_source, self).step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return super(CartPoleEnv_source, self).reset()