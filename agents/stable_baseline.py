import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make('Acrobot-v1')

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=6.3e-4, policy_kwargs = dict(net_arch=[256, 256]), batch_size= 128,
  buffer_size = 50000,
  learning_starts = 0,
  gamma= 0.99,
  train_freq = 4)

# Train the agent
model.learn(total_timesteps=int(1e5))
# Save the agent
model.save("dqn_acrobot")

"""
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("dqn_lunar")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

"""