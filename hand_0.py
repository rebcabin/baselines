import gym
env = gym.make("TwoHandsManipulateBlocks-v0")
# env = gym.make("HandManipulateBlock-v0")
# env = gym.make("CartPole-v1")
# env = gym.make("Zaxxon-v0")

observation = env.reset()
for _ in range(50):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  if done:
    observation = env.reset()
env.close()
