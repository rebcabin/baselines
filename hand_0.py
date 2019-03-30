import gym
env = gym.make("TwoHandsManipulateBlocks-v0")
# env = gym.make("HandManipulateBlock-v0")

# original was "HandManipulateBlock{}-v0, which is registered to
# <-- gym.envs.robotics.HandBlockEnv in manipulate.py
#
# "TwoHandsManipulateBlocks-v0" started as a plumbed copy, registered to
# <-- gym.envs.robotics:TwoHandsBlockEnvBBeckman in manipulate.py
#
# Search this repository for "bbeckman" to see my changes all over the place.

# env = gym.make("HandManipulateBlock-v0")
# env = gym.make("CartPole-v1")
# env = gym.make("Zaxxon-v0")

observation = env.reset()
for _ in range(2500):
    # through core.py::Wrapper.render,
    # hand_env.py::HandEnv.render
    # robot_env.py::RobotEnv.render
    env.render()
    action = env.action_space.sample() # your agent here
    # (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()
