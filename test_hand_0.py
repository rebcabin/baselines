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


def deprecated_test_vec_from_mat():
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    vec = np.reshape(mat, (6,))
    temp0 = vec == np.array([1, 2, 3, 4, 5, 6])
    result = np.all(temp0)
    assert result


def deprecated_test_mat_from_vec():
    vec = np.array([1, 2, 3, 4, 5, 6])
    mat = np.reshape(vec, (2, 3))
    temp0 = mat == np.array([[1, 2, 3], [4, 5, 6]])
    result = np.all(temp0)
    assert result


def sample_d_ball_method_1(d=LRV_DIM, n=10):
    """5,000 times slower than method 2, but generalizable to arbitrary
    ellipsoidal covariances."""
    mu = np.zeros(d, dtype=np.float32)

    # Sample multinormal in N+2 dims, then throw away any two dims.
    # the result will be uniform in N dims (see Robotics tech report
    # 18).

    # Even though shape checks out, np.random doesn't like the sparse matrix.
    # https://stackoverflow.com/questions/55503057/. My attempt at frugality
    # fails:

    # cov = sparse.identity(dim, dtype=np.float32)

    cov = np.identity(d, dtype=np.float32)
    assert (d, d) == cov.shape
    result = []
    for _ in range(n):
        temp0 = np.random.multivariate_normal(mu, cov)
        # intentionally raise div-by-zero if denominator *is* zero-vector
        temp2 = np.linalg.norm(temp0)
        temp1 = temp0 / temp2
        result.append(temp1[2:])
    return result


def sample_d_ball_method_2(d=LRV_DIM, n=10):
    """Direct implementation of the 'hat-box' method that just throws
    away two coordinates, exploiting Archimedes' hat-box theorem of 400 BC or
    thereabouts. This version doesn't use the 'nengo' library. It's a tiny bit
    faster than method 4."""
    temp0 = np.random.randn(n, d + 2)
    temp1 = temp0 / np.linalg.norm(temp0, axis=1, keepdims=True)
    result = temp1[:, 2:]
    return result


def sample_d_ball_method_3(d=LRV_DIM, n=10):
    """This method replaces radius with a distribution uniform in r^(1/d).
    That's more risky and 15 percent slower when the number of dimensions
    gets large (e.g., 1220, as in our case)."""
    result = nengo.dists.get_samples(
        nengo.dists.UniformHypersphere(surface=False), n=n, d=d)
    return result


def sample_d_ball_method_4(d=LRV_DIM, n=10):
    """This is the 'hat-box' method that simply throws away two coordinates,
    exploiting Archimedes' hat-box theorem of 400 BC or thereabouts. This
    uses the nengo library and is a little slower than method two. """
    temp = nengo.dists.get_samples(
        nengo.dists.UniformHypersphere(surface=True), n=n, d=d + 2)
    result = temp[:, 2:]
    return result


SAMPLE_D_BALL_METHODS = [None,
                         sample_d_ball_method_1, sample_d_ball_method_2,
                         sample_d_ball_method_3, sample_d_ball_method_4]


def deprecated_test_plot_spherically_uniform_lrvs(method=2):
    """Visually shows that lrvs drawn from the methods above are uniform. Not
    as persuasive as a chi-square test, but good enough for engineering."""
    n = 10000
    d = LRV_DIM
    points = SAMPLE_D_BALL_METHODS[method](d=d, n=n)
    assert points.shape == (n, d)
    radii = np.linalg.norm(points, axis=1, keepdims=True)
    radius_powers = radii ** d
    plt.hist(radius_powers)
    plt.show()  # press ctrl-w to close the window
    assert True


def deprecated_test_lrv_gen_speeds():
    """Shows that method2 is the fastest. Reactivate if you want to verify."""
    a_dict = {}
    for i in range(1, len(SAMPLE_D_BALL_METHODS)):
        a_dict[i] = timeit.timeit(SAMPLE_D_BALL_METHODS[i], number=1)
    print(a_dict)


def new_zero_lrm():
    return np.zeros(LRM_SHAPE)


def new_uniformly_random_lrv():
    result = sample_d_ball_method_2(d=LRV_DIM, n=1)
    return result


def new_uniformly_random_lrm(sigma=1.0):
    temp0 = new_uniformly_random_lrv()
    temp1 = temp0[0]
    if sigma != 1.0:
        temp2 = temp1 * sigma
    else:
        temp2 = temp1
    result = lrm_from_lrv(temp2)
    return result


LRM_EMPIRICAL_SCALE_FACTOR_HYPERPARAMETER = 2.0
LRM_SHRINKING_SIGMA = 1.0
LRM_SHRINKING_FACTOR_HYPERPARAMETER = 0.992


def lrm_from_states_and_time_step(left_side, rigt_side, time_step):
    _ignore_for_now = time_step
    left_lrm = new_uniformly_random_lrm(1.0) * LRM_EMPIRICAL_SCALE_FACTOR_HYPERPARAMETER
    rigt_lrm = new_uniformly_random_lrm(1.0) * LRM_EMPIRICAL_SCALE_FACTOR_HYPERPARAMETER
    left_action = np.dot(left_lrm, left_side)
    rigt_action = np.dot(rigt_lrm, rigt_side)
    action = np.concatenate([left_action, rigt_action])
    return action


def test_hands():
    sigma = 1.0
    action = env.action_space.sample()  # your agent here
    temp = new_uniformly_random_lrm(1.0)
    for time_step in range(250):
        # through core.py::Wrapper.render,
        # hand_env.py::HandEnv.render
        # robot_env.py::RobotEnv.render
        env.render()
        # (the above takes random actions)
        observation, reward, done, info = env.step(action)
        inspect_me_in_debugger = env.observation_space
        left_side = observation['left_side']
        rigt_side = observation['rigt_side']
        action = lrm_from_states_and_time_step(left_side, rigt_side, time_step)
        if done:
            _ = env.reset()
    env.close()
