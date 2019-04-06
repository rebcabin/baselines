import nengo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from collections import namedtuple as ntup
from multiprocessing import Pool


D_BALL_RADIUS = 1.0


def sample_d_ball(d=2, n=300):
    return nengo.dists.get_samples(
        nengo.dists.UniformHypersphere(surface=False), n=n, d=d)


def get_target(dim):
    seek_target = sample_d_ball(d=dim, n=1)
    # # Check the sample.
    # assert seek_target[0].shape == (dim,)
    # for point in seek_target:
    #     assert point[0] >= -1 and point[0] <= 1
    return seek_target[0]


def sample_a_b(dim, center, sigma, target):
    assert center.shape == (dim,)
    assert target.shape == (dim,)

    y1 = center + sigma * np.random.randn(dim)
    y2 = center + sigma * np.random.randn(dim)

    assert y1.shape == (dim,)
    assert y2.shape == (dim,)

    # Consult the oracle (ground truth).
    d1 = distance.euclidean(target, y1)
    d2 = distance.euclidean(target, y2)

    yp = y1 if d1 < d2 else y2
    dp = min(d1, d2)  # This is the CHEATING port of the oracle.

    return yp, dp


def seeker_with_max_experiment(
        dimensions=4,
        rollouts=10,
        maxsamples=10000,
        allowederror=0.1,
        covariance_decay=0.992):
    # Per-experiment local variables:
    convergefailures = 0
    convergesuccesses = 0
    convergetotaltrials = 0
    targets = [0] * rollouts
    convergestepss = [0] * rollouts
    convergesigmas = [0] * rollouts
    convergedistances = [0] * rollouts
    convergemaxdistances = [0] * rollouts
    success_p = [False] * rollouts

    for rollout in range(rollouts):
        # Choose a random ground-truth uniformly inside the d-ball.
        seek_target = get_target(dimensions)
        count = 0
        yp = np.zeros(dimensions)  # first guess is the origin
        maxdist = -1000000
        totalbigdist = 0
        totalsmldist = 0
        dimdist = D_BALL_RADIUS  ## usually 1.0
        sigma = (dimdist ** 2)
        sigma_0 = sigma
        calcdistance = 0

        while (distance.euclidean(seek_target, yp) > \
               allowederror) and (count < maxsamples):
            # Generate new guesses centered at the current best guess yp.
            yp, calcdistance = sample_a_b(dimensions, yp, sigma, seek_target)
            count += 1
            sigma *= covariance_decay
            # Maintain statistics for this rollout.
            if calcdistance < D_BALL_RADIUS:
                totalsmldist += 1
            else:
                totalbigdist += 1
            maxdist = max(maxdist, calcdistance)

        if count < maxsamples:
            success_p[rollout] = True

        targets[rollout] = seek_target
        convergestepss[rollout] = count
        convergesigmas[rollout] = sigma
        convergedistances[rollout] = calcdistance
        convergemaxdistances[rollout] = maxdist

        if (count >= maxsamples):
            convergefailures += 1
        else:
            convergesuccesses += 1
            convergetotaltrials += count

        vseeker = ntup('VogelsongSeeker',
                       ['rollouts', 'failures', 'successes',
                        'targets',
                        'mean_trials_per_success',
                        'percentage_successful',
                        'big_distance_count',
                        'small_distance_count',
                        'original_sigma',
                        'covariance_decay_per_trial',
                        'convergestepss', 'convergesigmas',
                        'convergedistances',
                        'convergemaxdistances',
                        'success_flags'
                        ])

    average_trials_per_success = (
        (convergetotaltrials * 1.0 / convergesuccesses)
        if convergesuccesses > 0 else 0)

    percentage_successful = (
            100.0 * (rollouts - convergefailures)
            / rollouts
    )

    result = vseeker(rollouts, convergefailures, convergesuccesses,
                     targets,
                     average_trials_per_success,
                     percentage_successful,
                     totalbigdist, totalsmldist,
                     sigma_0,
                     covariance_decay,
                     convergestepss, convergesigmas,
                     convergedistances,
                     convergemaxdistances,
                     success_p)

    return result


def g_analyze_seeker_with_max_experiment(expt):
    std_dev_trials_per_success = np.std(
        [k[1] for k in zip(expt.success_flags,
                           expt.convergestepss)
              if k[0]])
    return {'average_trials_per_success': expt.mean_trials_per_success,
            'percentage_successful': expt.percentage_successful,
            'rollouts': expt.rollouts,
            'covariance_decay': expt.covariance_decay_per_trial,
            'std_dev_trials_per_success': std_dev_trials_per_success}


# dim_11_experiments = [analyze_seeker_with_max_experiment(
#     seeker_with_max_experiment(
#         dimensions=27,
#         rollouts=100,
#         covariance_decay=cd
#     )) for cd in np.linspace(0.990, 0.999, 10)]


def expg(cd):
    return g_analyze_seeker_with_max_experiment(
        seeker_with_max_experiment(
            dimensions=260,
            maxsamples=80000,
            rollouts=100,
            covariance_decay=cd))


with Pool(10) as p:
    dim_11_experiments = p.map(
        expg,
        list(np.linspace(0.9980, 0.9999, 10))
    )


dim_11_experiments_df = pd.DataFrame(dim_11_experiments)


dim_11_experiments_df.to_csv("dim_11_vogelsong_df")


dim_11_experiments_df = pd.DataFrame.from_csv("dim_11_vogelsong_df")


fig, ax = plt.subplots()
ax.errorbar(
    dim_11_experiments_df['covariance_decay'],
    dim_11_experiments_df['average_trials_per_success'],
    yerr=dim_11_experiments_df['std_dev_trials_per_success'],
    label='average number of trials per success'
)
ax.plot(dim_11_experiments_df['covariance_decay'],
        400 * dim_11_experiments_df['percentage_successful'],
        label='percentage successful (times 2 for scale)')
ax.set(xlabel="covariance decay",
       ylabel="average number of trials to succeed",
       title="tradeoff between success percentage and number of trials\n11 dimension, 100 rollouts per data point")
ax.grid()
ax.legend()
plt.show()

