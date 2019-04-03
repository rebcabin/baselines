import gym
import numpy as np
import scipy as sp
import scipy.sparse as sparse

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


# Abbreviations:
# LRM :: Linear Regulator Matrix, #actions rows x #states columns
# LRV :: Linear Regulator Vector,


ONE_HAND_STATE_DIM = 48
ONE_CUBE_STATE_DIM = 13
ONE_SIDE_STATE_DIM = ONE_HAND_STATE_DIM + ONE_CUBE_STATE_DIM
ONE_HAND_ACTION_DIM = 20
LRV_DIM = ONE_SIDE_STATE_DIM * ONE_HAND_ACTION_DIM


LRM_SHAPE = (ONE_HAND_ACTION_DIM, ONE_SIDE_STATE_DIM)
LRV_SHAPE = (LRV_DIM,)


def lrm_from_lrv(vec):
    assert vec.shape == LRV_SHAPE
    result = np.reshape(vec, LRM_SHAPE)
    return result


def lrv_from_lrm(mat):
    assert mat.shape == LRM_SHAPE
    result = np.reshape(mat, LRV_SHAPE)
    return result


def test_vec_from_mat():
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    vec = np.reshape(mat, (6,))
    temp0 = vec == np.array([1, 2, 3, 4, 5, 6])
    result = np.all(temp0)
    assert result


def test_mat_from_vec():
    vec = np.array([1, 2, 3, 4, 5, 6])
    mat = np.reshape(vec, (2, 3))
    temp0 = mat == np.array([[1, 2, 3], [4, 5, 6]])
    result = np.all(temp0)
    assert result


def new_spherical_LRV(mu, sigma):
    assert mu.shape == LRV_SHAPE
    # cov = sparse.identity(LRV_DIM, dtype=np.float32)
    # cov = np.array(LRM_SHAPE, dtype=np.float32)
    cov = np.array((42, 42), dtype=np.float32)
    assert cov.shape == (42, 42)
    assert LRM_SHAPE == (20, 61)
    # git problem?
    assert cov.shape == (20, 61)
    # assert cov.shape == LRM_SHAPE
    # for i in range(LRV_DIM):
    #     cov.itemset((i, i), 1.0)
    # assert cov.shape == LRV_DIM
    return cov
    # result = np.random.multivariate_normal(
    #     mu, sigma * sparse.identity(LRV_DIM))
    # return result


def test_new_spherical_LRV():
    temp0 = new_spherical_LRV(
        np.zeros(LRV_SHAPE),
        1.0
    )


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
