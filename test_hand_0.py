import gym
import numpy as np
import matplotlib.pyplot as plt
import nengo
import timeit

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


def new_spherically_uniform_LRV(sigma_0):
    dim = LRV_DIM + 2
    mu = np.zeros(dim, dtype=np.float32)
    # Sample multinormal in N+2 dims, then throw away any two dims.
    # the result will be uniform in N dims (see Robotics tech report
    # 18).

    # Even though shape checks out, np.random doesn't like the sparse matrix.
    # https://stackoverflow.com/questions/55503057/. My attempt at frugality
    # fails
    # cov = sigma_0 * sparse.identity(dim, dtype=np.float32)
    cov = sigma_0 * np.identity(dim, dtype=np.float32)
    assert (dim, dim) == cov.shape
    temp0 = np.random.multivariate_normal(mu, cov)
    # intentionally raise div-by-zero if denominator *is* zero-vector
    temp1 = temp0 / np.linalg.norm(temp0)
    result = temp1[2:]
    return result


def test_new_spherically_uniform_LRV():
    temp0 = new_spherically_uniform_LRV(1.0)
    assert temp0.shape == LRV_SHAPE


def sample_d_ball_method_1(n_points):
    sigma = 1.0
    index1 = np.random.randint(0, LRV_DIM + 2)
    index2 = np.random.randint(0, LRV_DIM + 2)
    while index1 == index2:
        index2 = np.random.randint(0, LRV_DIM + 2)
    temp = [(p[index1], p[index2])
            for p in [new_spherically_uniform_LRV(sigma)
                      for _ in range(n_points)]]
    points = [p / np.linalg.norm(p) for p in temp][2:]
    return points


def sample_d_ball_method_2(n_points):
    sigma = 1.0
    index1 = np.random.randint(0, LRV_DIM + 2)
    index2 = np.random.randint(0, LRV_DIM + 2)
    while index1 == index2:
        index2 = np.random.randint(0, LRV_DIM + 2)
    temp = []
    for _ in range(n_points):
        p = new_spherically_uniform_LRV(sigma)
        temp.append((p[index1], p[index2]))
    points = [p / np.linalg.norm(p) for p in temp][2:]
    return points


# prove that method3 is fastest


def sample_d_ball_method_3(d=LRV_DIM, n=10):
    result = nengo.dists.get_samples(
        nengo.dists.UniformHypersphere(surface=False), n=n, d=d)
    return result


def sample_d_ball_method_4(d=LRV_DIM, n=10):
    temp = nengo.dists.get_samples(
        nengo.dists.UniformHypersphere(surface=True), n=n, d=d + 2)
    result = temp[:, 2:]
    return result


# print(timeit.timeit("sample_d_ball_method_1(10)",
#                     setup="from test_hand_0 import sample_d_ball_method_1",
#                     number=1))
#
# print(timeit.timeit("sample_d_ball_method_2(10)",
#                     setup="from test_hand_0  import sample_d_ball_method_2",
#                     number=1))
#
print(timeit.timeit("sample_d_ball_method_3(d=10)",
                    setup="from test_hand_0  import sample_d_ball_method_3",
                    number=1))


def get_distinct_random_vector_indices(d=LRV_DIM):
    index1 = np.random.randint(0, d)
    index2 = np.random.randint(0, d)
    while index1 == index2:
        index2 = np.random.randint(0, d)
    return index1, index2


def test_plot_spherically_uniform_LRVs():
    n_points = 1000
    d = 1220
    # sigma = 1.0
    index1, index2 = get_distinct_random_vector_indices(d)
    plottable_points = []
    points = sample_d_ball_method_4(d=d, n=n_points)
    assert points.shape == (n_points, d)
    for i in range(n_points):
        p = points[i]
        plottable_points.append((p[index1], p[index2]))
    fix, ax = plt.subplots()
    ax.scatter(*np.transpose(plottable_points))
    ax.set_aspect(1.0)
    plt.show()
    result = points
    assert True


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
