# %%
from jax import numpy as np
from jax import random
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
from jifty1 import hmc
import scipy
from scipy.special import comb

# %% converting moments helpers from
# https://www.statsmodels.org/stable/_modules/statsmodels/stats/moment_helpers.html

def _convert_to_multidim(x):
    if any([isinstance(x, list), isinstance(x, tuple)]):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        # something strange was passed and the function probably
        # will fall, maybe insert an exception?
        return x


def _convert_from_multidim(x, totype=list):
    if len(x.shape) < 2:
        return totype(x)
    return x.T

def mnc2mc(mnc, wmean=True):
    """convert non-central to central moments, uses recursive formula
    optionally adjusts first moment to return mean
    """
    X = _convert_to_multidim(mnc)

    def _local_counts(mnc):
        mean = mnc[0]
        mnc = [1] + list(mnc)  # add zero moment = 1
        mu = []
        for n, m in enumerate(mnc):
            mu.append(0)
            for k in range(n + 1):
                sgn_comb = (-1) ** (n - k) * comb(n, k, exact=True)
                mu[n] += sgn_comb * mnc[k] * mean ** (n - k)
        if wmean:
            mu[1] = mean
        return mu[1:]

    res = np.apply_along_axis(_local_counts, 0, X)
    # for backward compatibility convert 1-dim output to list/tuple
    return res


# %%
dists = [
    stats.bernoulli,
#    stats.beta,
    stats.betabinom,
    stats.cauchy,
    stats.chi2,
    stats.dirichlet,
    stats.expon,
    stats.gamma,
    stats.geom,
    stats.laplace,
    stats.logistic,
    stats.multivariate_normal,
    stats.norm,
    stats.pareto,
    stats.poisson,
    stats.t,
    stats.uniform
]

# continuous distributions
dists = list(filter(lambda d: hasattr(d, 'logpdf'), dists))

# but only want these because they have no extra parameters
# cauchy excluded because of bad plots due to fat tails
dists =  [stats.cauchy, stats.expon, stats.laplace, stats.logistic, stats.norm]
# %%
import importlib

def sample_and_plot(distribution):
    distribution = importlib.import_module(distribution)

    name = distribution.__name__.split('.')[-1]

    print(name)

    sampler = hmc.NUTSChain(
        initial_position = np.array(1.03890),
        potential_energy = lambda x: -1 * distribution.logpdf(x),
        diag_mass_matrix = 1.,
        eps = 0.7193,
        maxdepth = 20,
        rngseed = 42,
        compile = True,
        dbg_info = True
    )

    _last_pos, _key, position_samples, unintegrated_momenta, momentum_samples, depths, trees = sampler.generate_n_samples(1000)

    (unique, counts) = np.unique(depths, return_counts=True)
    depths_frequencies = np.asarray((unique, counts)).T

    xs = np.linspace(-10, 10)

    if name == 'expon':
        xs = np.linspace(0, 10)

    bins = xs

    plt.hist(position_samples, bins=bins, density=True)
    plt.plot(xs, distribution.pdf(xs), color='r')
    plt.title(name)
    plt.show()

    # central moments
    sample_moms_central = scipy.stats.moment(position_samples, [1,2,3,4,5,6])
    # include mean i.e. make first moment non central
    sample_moms_central[0] = np.mean(position_samples)
    # these are non central
    dist_moms_non_central = np.array([eval('scipy.stats.' + name + '.moment(i)') for i in [1,2,3,4,5,6]])
    # convert to central moments, again include mean
    dist_moms_central = mnc2mc(dist_moms_non_central, wmean = True)

    return name, depths_frequencies, sample_moms_central, np.array(dist_moms_central)

# %%
# parallel
from multiprocessing import Pool

pool = Pool()

name_and_moms_list = pool.map(sample_and_plot, list(map(lambda d: d.__name__, dists)))
#name_and_moms_list = [pool.apply_async(sample_and_plot, d.__name__) for d in dists]

# %%
moms_dict = {key: (sample_moms, dist_moms) for key, _, sample_moms, dist_moms in name_and_moms_list}
depths_dict = {key: depth for key, depth, _, _ in name_and_moms_list}

# %%
mom_absolute_differences = {k: np.abs(s - d) for k, (s, d) in moms_dict.items()}
# difference over average magnitude
mom_relative_differences = {k: 2 * np.abs(s - d) / (np.abs(d) + np.abs(s)) for k, (s, d) in moms_dict.items()}

from pprint import pprint
print("depths")
pprint(depths_dict)
print("moments")
pprint(moms_dict)
print("abosulte differences")
pprint(mom_absolute_differences)
print("relative differences")
pprint(mom_relative_differences)

# %%
# sequential
#name_and_moms_list = []
#for d in dists:
#    name_and_moms_list.append(sample_and_plot(d.__name__))
#
# %%
