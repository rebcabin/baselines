import gym
env = gym.make("TwoHandsManipulateBlocks-v0")  # manipulate.py
# env = gym.make("HandManipulateBlock-v0")

# [[[ bbeckman:
# original was "HandManipulateBlock{}-v0, which is registered to
# <-- gym.envs.robotics.HandBlockEnv, which happens to be in manipulate.py.
# The registration code is in envs/__init__.py, line 407.
#
# "TwoHandsManipulateBlocks-v0" started as a plumbed copy, registered to
# <-- gym.envs.robotics:TwoHandsBlockEnvBBeckman, which happens to be in
# manipulate.py. The registration code is in envs/__init__.py, line 415.
# The registration code specifies some initial conditions that may have
# to change (TODO).
#
# Search this repository for "bbeckman" to see my changes all over the place.
# ]]]

# env = gym.make("HandManipulateBlock-v0")
# env = gym.make("CartPole-v1")
# env = gym.make("Zaxxon-v0")

_ = env.reset()
for _ in range(250):
    # through core.py::Wrapper.render,
    # hand_env.py::HandEnv.render
    # robot_env.py::RobotEnv.render
    env.render()
    action = env.action_space.sample()  # your agent here
    # (this takes random actions)
    observation, reward, done, info = env.step(action)
    _ = env.observation_space
    if done:
        _ = env.reset()
    # [[[ bbeckman: inspect the following in the debugger ]]]
    observation_space = env.observation_space
env.close()
