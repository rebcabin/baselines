import gym
import numpy as np
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

ONE_HAND_STATE_DIM = 48
ONE_CUBE_STATE_DIM = 13
ONE_SIDE_STATE_DIM = ONE_HAND_STATE_DIM + ONE_CUBE_STATE_DIM
ONE_HAND_ACTION_DIM = 20
ONE_LINEAR_REGULATOR_DIM = ONE_SIDE_STATE_DIM * ONE_HAND_ACTION_DIM


def lin_reg_matrix_from_lin_reg_vector(vec, new_shape):
    pass


def lin_reg_vector_from_lin_reg_matrix(mat, new_shape):
    pass


def test_vec_from_mat():
    mat = np.array([[1, 2, 3],
                    [4, 5, 6]])
    vec = np.reshape(mat, (6,))
    temp0 = vec == np.array([1, 2, 3, 4, 5, 6])
    result = np.all(temp0)
    assert result


def test_mat_from_vec():
    vec = np.array([1, 2, 3, 4, 5, 6])
    mat = np.reshape(vec, (2, 3))
    temp0 = mat == np.array([[1, 2, 3],
                             [4, 5, 6]])
    result = np.all(temp0)
    assert result


def new_random_linear_regulator(sigma):
    result = np.random
    return result


def test_pytest_itself():
    assert True


state_left = np.zeros([20, 61])
state_rigt = np.zeros([20, 61])

for _ in range(25):
    # through core.py::Wrapper.render,
    # hand_env.py::HandEnv.render
    # robot_env.py::RobotEnv.render
    env.render()
    action = env.action_space.sample()  # your agent here
    # (this takes random actions)
    observation, reward, done, info = env.step(action)
    inspect_me_in_debugger = env.observation_space
    if done:
        _ = env.reset()

env.close()
