import gym
import numpy as np
import matplotlib.pyplot as plt
import timeit
import pytest


#  ___       __ _      _ _   _
# |   \ ___ / _(_)_ _ (_) |_(_)___ _ _  ___
# | |) / -_)  _| | ' \| |  _| / _ \ ' \(_-<
# |___/\___|_| |_|_||_|_|\__|_\___/_||_/__/


# LRM :: Linear Regulator Matrix, num-actions rows x num-states columns
# LRV :: Linear Regulator Vector,
# STATE :: POSITION concat VELOCITY
# DOF : :: degrees of freedom
# POSITION :: 3D position in the world coordinate frame
# VELOCITY :: time derivative of position
# ORIENTATION :: 4D normalized quaternion --- three degrees of freedom
#              (a normalized quaternion is also called a "versor.")
# ANGULAR_VELOCITY :: 3D vector measured in radians per second
# POSE :: 7D, 6-DOF POSITION concat ORIENTATION
# POSE_VELOCITY :: 6D, VELOCITY concat ANGULAR_VELOCITY


#   ___              ___         _                            _
#  / __|_  _ _ __   | __|_ ___ _(_)_ _ ___ _ _  _ __  ___ _ _| |_
# | (_ | || | '  \  | _|| ' \ V / | '_/ _ \ ' \| '  \/ -_) ' \  _|
#  \___|\_, |_|_|_| |___|_||_\_/|_|_| \___/_||_|_|_|_\___|_||_\__|
#       |__/


# The code for this environment is TwoHandsBlockEnvBBeckman in manipulate.py.
# The registration code is at line 415 (or search for "bbeckman") in
# gym/envs/__init__.py. The registration code maps the string
# "TwoHandsManipulateBlocks-v0" to the class 'TwoHandsBlockEnvBBeckman.'


env = gym.make("TwoHandsManipulateBlocks-v0")


env_type = type(env)


_ = env.reset()


# inspect this type in the debugger; it's a time-limit wrapper!

# env = gym.make("HandManipulateBlock-v0")

# [[[ bbeckman:
# original was "HandManipulateBlock{}-v0, which is registered to
# <-- gym.envs.robotics.HandBlockEnv, which happens to be in manipulate.py.
# The registration code is in envs/__init__.py, line 407.
#
# "TwoHandsManipulateBlocks-v0" started as a plumbed copy, registered to <--
# gym.envs.robotics:TwoHandsBlockEnvBBeckman, in manipulate.py. The
# registration code is in envs/__init__.py, line 415. The registration code
# specifies initial conditions that may have to change (TODO).
#
# Search this repository for "bbeckman" to see my changes all over the place.
# ]]]


# Many functions assign a value to a variable named "result" and then
# immediately return "result." The convention makes it easy to inspect the
# result in the debugger by setting a breakpoint on "return result."


#  ___     _ _      _     _     _     ___ _        _
# | __|  _| | |  _ | |___(_)_ _| |_  / __| |_ __ _| |_ ___
# | _| || | | | | || / _ \ | ' \  _| \__ \  _/ _` |  _/ -_)
# |_| \_,_|_|_|  \__/\___/_|_||_\__| |___/\__\__,_|\__\___|


ONE_HAND_STATE_DIM = 48
ONE_CUBE_POSE_DIM = 7
ONE_CUBE_VELOCITY_DIM = 6
ONE_CUBE_STATE_DIM = ONE_CUBE_POSE_DIM + ONE_CUBE_VELOCITY_DIM
ONE_SIDE_STATE_DIM = ONE_HAND_STATE_DIM + ONE_CUBE_STATE_DIM
ONE_HAND_ACTION_DIM = 20
LRV_DIM = ONE_SIDE_STATE_DIM * ONE_HAND_ACTION_DIM

LRM_SHAPE = (ONE_HAND_ACTION_DIM, ONE_CUBE_POSE_DIM)
LRV_SHAPE = (LRV_DIM,)


def lrm_from_lrv(vec):
    assert vec.shape == LRV_SHAPE
    result = np.reshape(vec, LRM_SHAPE)
    return result


def lrv_from_lrm(mat):
    assert mat.shape == LRM_SHAPE
    result = np.reshape(mat, LRV_SHAPE)
    return result


def new_zero_lrm():
    return np.zeros(LRM_SHAPE)


def new_normally_distributed_lrm(center, sigma):
    assert center.shape == LRM_SHAPE
    temp0 = np.random.randn(*LRM_SHAPE)
    assert temp0.shape == LRM_SHAPE
    result = sigma * temp0 + center
    return result


#  ___ _                   _   _        ___        _ _   _
# | __(_)_ _  __ _ ___ _ _| |_(_)_ __  | _ \___ __(_) |_(_)___ _ _
# | _|| | ' \/ _` / -_) '_|  _| | '_ \ |  _/ _ (_-< |  _| / _ \ ' \
# |_| |_|_||_\__, \___|_|  \__|_| .__/ |_| \___/__/_|\__|_\___/_||_|
#            |___/              |_|


ONE_HAND_FINGERTIP_COUNT = 5
POSITION_DIM = 3
ONE_HAND_FINGERTIP_POSITION_DIM = ONE_HAND_FINGERTIP_COUNT * POSITION_DIM
# including velocities:
ONE_HAND_FINGERTIP_STATE_DIM = ONE_HAND_FINGERTIP_POSITION_DIM * 2

# shorter names
LRV_FINGERTIP_DIM = \
    ONE_HAND_FINGERTIP_POSITION_DIM * ONE_HAND_ACTION_DIM

LRM_FINGERTIP_SHAPE = (ONE_HAND_ACTION_DIM, ONE_HAND_FINGERTIP_POSITION_DIM)
LRV_FINGERTIP_SHAPE = (LRV_FINGERTIP_DIM,)


def lrm_fingertip_from_lrv_fingertip(vec):
    assert vec.shape == LRV_FINGERTIP_SHAPE
    result = np.reshape(vec, LRM_FINGERTIP_SHAPE)
    return result


def lrv_fingertip_from_lrm_fingertip(mat):
    assert mat.shape == LRM_FINGERTIP_SHAPE
    result = np.reshape(mat, LRV_FINGERTIP_SHAPE)
    return result


def new_zero_lrm_fingertip():
    return np.zeros(LRM_FINGERTIP_SHAPE)


def new_normally_distributed_lrm_fingertip(center, sigma):
    assert center.shape == LRM_FINGERTIP_SHAPE
    temp0 = np.random.randn(*LRM_FINGERTIP_SHAPE)
    assert temp0.shape == LRM_FINGERTIP_SHAPE
    result = sigma * temp0 + center
    return result


#  ___                  _ _
# / __| __ _ _ __  _ __| (_)_ _  __ _
# \__ \/ _` | '  \| '_ \ | | ' \/ _` |
# |___/\__,_|_|_|_| .__/_|_|_||_\__, |
#                 |_|           |___/


def sample_d_ball_uniformly(d=LRV_DIM, n=10):
    """Discard any two coordinates, exploiting Archimedes' hat-box
    theorem. Found in separate experiments to be faster than "nengo"
    and three orders of magnitude faster than multivariate Gaussian,
    which requires a covariance matrix. """
    temp0 = np.random.randn(n, d + 2)
    temp1 = temp0 / np.linalg.norm(temp0, axis=1, keepdims=True)
    result = temp1[:, 2:]
    return result


#  _  _                                                  _
# | || |_  _ _ __  ___ _ _ _ __  __ _ _ _ __ _ _ __  ___| |_ ___ _ _ ___
# | __ | || | '_ \/ -_) '_| '_ \/ _` | '_/ _` | '  \/ -_)  _/ -_) '_(_-<
# |_||_|\_, | .__/\___|_| | .__/\__,_|_| \__,_|_|_|_\___|\__\___|_| /__/
#       |__/|_|           |_|


# TODO: violation code-review guideline 24: move to config file!

# Modify the following factor so that the hands actually turn the
# cube but don't drop it too often.

LRM_EMPIRICAL_SCALE_FACTOR_HYPERPARAMETER = 1.250

# TODO: Consider one scale factor per finger.
LRM_SHRINKING_SIGMA = 1.0
LRM_SIGMA_SHRINKING_FACTOR_HYPERPARAMETER = 0.9998
LEFT = 0
RIGT = 1
LRMS_LIFESPAN_IN_TIME_STEPS_HYPERPARAMETER = 10
TRIAL_LIFESPAN_IN_TIME_STEPS = 250


#  ___ _                   _           _    _
# | _ (_)___ __ _____ __ _(_)___ ___  | |  (_)_ _  ___ __ _ _ _
# |  _/ / -_) _/ -_) V  V / (_-</ -_) | |__| | ' \/ -_) _` | '_|
# |_| |_\___\__\___|\_/\_/|_/__/\___| |____|_|_||_\___\__,_|_|
#    _      _   _
#   /_\  __| |_(_)___ _ _  ___
#  / _ \/ _|  _| / _ \ ' \(_-<
# /_/ \_\__|\__|_\___/_||_/__/



# [[[ bbeckman: under the scheme of PLAs (piecewise-linear actions), action is
# a piecewise linear transformation of the residual between the desired
# configuration (called 'goal') of the cube and the actual configuration
# (called 'achieved_goal'). The residual is a 7-vector: three positions, three
# angles expressed as a 4D normalized quaternion. Ignore velocities for now
# (TODO); they are an additional 6-vector. The action-dim is 20. The linear
# transformation is, therefore, a 20 x 7 matrix. There is one such matrix every
# so many time steps, where 'so many' is
# LRMS_LIFESPAN_IN_TIME_STEPS_HYPERPARAMETER. The number of such matrices in
# the lifetime of a trial is the following: ]]]


LRMSS_LENGTH = TRIAL_LIFESPAN_IN_TIME_STEPS // \
               LRMS_LIFESPAN_IN_TIME_STEPS_HYPERPARAMETER


def starting_lrmss(sigma):
    result = [starting_lrms(sigma) for _ in range(LRMSS_LENGTH)]
    return result


def evolved_lrms(center, sigma):
    result = [new_normally_distributed_lrm(center, sigma) *
              LRM_EMPIRICAL_SCALE_FACTOR_HYPERPARAMETER,

              new_normally_distributed_lrm(center, sigma) *
              LRM_EMPIRICAL_SCALE_FACTOR_HYPERPARAMETER]
    return result


def starting_lrms(sigma):
    return evolved_lrms(new_zero_lrm(), sigma)


def evolved_lrmss(left_or_right, lrmss, sigma):
    result = [evolved_lrms(lrmss[i][left_or_right], sigma)
              for i in range(LRMSS_LENGTH)]
    return result


def get_lrms(sequence, time_step):
    result = sequence[time_step // LRMS_LIFESPAN_IN_TIME_STEPS_HYPERPARAMETER]
    return result


def action_from_state(lrms, state, t):
    left_side = state['left_residual']
    rigt_side = state['rigt_residual']
    left_lrm = lrms[LEFT]
    rigt_lrm = lrms[RIGT]
    _ignore_for_now = t
    left_action = np.dot(left_lrm, left_side)
    rigt_action = np.dot(rigt_lrm, rigt_side)
    action = np.concatenate([left_action, rigt_action])
    return action


def collect_preference(state):
    # TODO: violation code-review guideline 12: code commented-out!
    # result = input('Express preference [a, l, b, r, n, q]:')

    # autograder
    a = state['left_loss']
    b = state['rigt_loss']

    result = 'a'
    if b < a:
        result = 'b'
    return result


class GameState(object):
    """TODO: UNDONE"""
    def run_hands():
        pass

    def record_output(self, c):

        output_dict = \
            {'y_chosen': list(self.yp),  # np.ndarray not json-serializable.
             'time_stamp':
             f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
             'left_guess': list(self.ys[0]),
             'right_guess': list(self.ys[1]),
             'disturbance_amplitude': self.amplitude,
             'repeatable_disturbance': self.repeatable_q,
             'truth': EXACT_LQR_CART_POLE_GAINS,
             'distance_from_left_guess_to_truth':
             distance.euclidean(self.ys[0], EXACT_LQR_CART_POLE_GAINS),
             'distance_from_right_guess_to_truth':
             distance.euclidean(self.ys[1], EXACT_LQR_CART_POLE_GAINS),
             'command': command_name(c, self.ground_truth_mode),
             'dimensions': self.sim_constants.dimensions,
             'sigma': np.sqrt(self.cov[0][0]),
             'trial_count': self.trial_count,
             'output_file_name': self.output_file_name}
        pp.pprint(output_dict)
        jsout = json.dumps(output_dict, indent=2)
        with open(self.output_file_name, "a") as output_file_pointer:
            print(jsout, file=output_file_pointer)
            time.sleep(1)


def test_hands():
    sigma = 1.0
    lrmss = starting_lrmss(sigma)
    action = action_from_state(lrmss[0], env.get_state(), t=-1)
    state = None
    while True:
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
        preference = collect_preference(state)
        if preference in ('a', 'l'):
            lrmss = evolved_lrmss(LEFT, lrmss, sigma)
            sigma *= LRM_SIGMA_SHRINKING_FACTOR_HYPERPARAMETER
        elif preference in ('b', 'r'):
            lrmss = evolved_lrmss(RIGT, lrmss, sigma)
            sigma *= LRM_SIGMA_SHRINKING_FACTOR_HYPERPARAMETER
        elif preference == 'q':
            break
        else:
            print('preference must be a, l, b, r, q, n')

    env.close()
