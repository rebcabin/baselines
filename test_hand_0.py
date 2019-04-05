import gym
import numpy as np
import matplotlib.pyplot as plt
import timeit
import pytest

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


SAMPLE_D_BALL_METHODS = [None,
                         sample_d_ball_method_1,
                         sample_d_ball_method_2]


@pytest.mark.skip(reason="unskip if you want to see distribution")
def test_plot_spherically_uniform_lrvs(method=2):
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


@pytest.mark.skip(reason="unskip if you want to see speeds")
def test_lrv_gen_speeds():
    """Shows that method2 is fastest. Reactivate if you want to verify."""
    a_dict = {}
    for i in range(1, len(SAMPLE_D_BALL_METHODS)):
        a_dict[i] = timeit.timeit(SAMPLE_D_BALL_METHODS[i], number=1)
    print(a_dict)


def new_zero_lrm():
    return np.zeros(LRM_SHAPE)


def new_uniformly_random_unit_radius_at_origin_lrv():
    result = sample_d_ball_method_2(d=LRV_DIM, n=1)
    return result


def new_uniform_lrm_at_origin(sigma=1.0):
    temp0 = new_uniformly_random_unit_radius_at_origin_lrv()
    temp1 = temp0[0]
    # premature optimization (violation of code-review guideline number 40):
    if sigma != 1.0:
        temp2 = temp1 * sigma
    else:
        temp2 = temp1
    result = lrm_from_lrv(temp2)
    return result


def new_normally_distributed_lrm(center, sigma):
    assert center.shape == LRM_SHAPE
    temp0 = np.random.randn(ONE_HAND_ACTION_DIM, ONE_SIDE_STATE_DIM)
    assert temp0.shape == LRM_SHAPE
    result = sigma * temp0 + center
    return result


LRM_EMPIRICAL_SCALE_FACTOR_HYPERPARAMETER = 1.5
LRM_SHRINKING_SIGMA = 1.0
LRM_SHRINKING_FACTOR_HYPERPARAMETER = 0.992
LEFT = 0
RIGT = 1
LRMS_LIFESPAN_IN_TIME_STEPS_HYPERPARAMETER = 10
TRIAL_LIFESPAN_IN_TIME_STEPS = 250


def starting_lrms(sigma):
    return [new_uniform_lrm_at_origin(sigma),
            new_uniform_lrm_at_origin(sigma)]


def starting_lrms_sequence(sigma):
    sequence_length = TRIAL_LIFESPAN_IN_TIME_STEPS // \
                      LRMS_LIFESPAN_IN_TIME_STEPS_HYPERPARAMETER
    result = [starting_lrms(sigma) for _ in range(sequence_length)]
    return result


def get_lrms(sequence, time_step):
    result = sequence[time_step // LRMS_LIFESPAN_IN_TIME_STEPS_HYPERPARAMETER]
    return result


def action_from_state(lrms, state, t):
    left_side = state['left_side']
    rigt_side = state['rigt_side']
    left_lrm = lrms[LEFT]
    rigt_lrm = lrms[RIGT]
    _ignore_for_now = t
    left_action = np.dot(left_lrm, left_side)
    rigt_action = np.dot(rigt_lrm, rigt_side)
    action = np.concatenate([left_action, rigt_action])
    return action


def test_hands():
    sigma = 1.0
    # Start with a random action from mujoco, just so we can get
    # action = env.action_space.sample()  # your agent here
    lrmss = starting_lrms_sequence(sigma)
    action = action_from_state(lrmss[0], env.get_state(), t=-1)
    for time_step in range(TRIAL_LIFESPAN_IN_TIME_STEPS):
        # through core.py::Wrapper.render,
        # hand_env.py::HandEnv.render
        # robot_env.py::RobotEnv.render
        env.render()
        state, reward, done, info = env.step(action)
        action = action_from_state(
            get_lrms(lrmss, time_step),
            state,
            time_step)
        if done:
            _ = env.reset()
    env.close()
