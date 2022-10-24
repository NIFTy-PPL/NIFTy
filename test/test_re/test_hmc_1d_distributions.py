import sys

import pytest
pytest.importorskip("jax")

from jax import numpy as jnp
from jax.scipy import stats
from numpy.testing import assert_allclose
import scipy
from scipy.special import comb

import nifty8.re as jft

pmp = pytest.mark.parametrize


def mnc2mc(mnc, wmean=True):
    """Convert non-central to central moments, uses recursive formula
    optionally adjusts first moment to return mean.
    """

    # https://www.statsmodels.org/stable/_modules/statsmodels/stats/moment_helpers.html
    def _local_counts(mnc):
        mean = mnc[0]
        mnc = [1] + list(mnc)  # add zero moment = 1
        mu = []
        for n, m in enumerate(mnc):
            mu.append(0)
            for k in range(n + 1):
                sgn_comb = (-1)**(n - k) * comb(n, k, exact=True)
                mu[n] += sgn_comb * mnc[k] * mean**(n - k)
        if wmean:
            mu[1] = mean
        return mu[1:]

    res = jnp.apply_along_axis(_local_counts, 0, mnc)
    # for backward compatibility convert 1-dim output to list/tuple
    return res


# Test simple distributions with no extra parameters
dists = [stats.cauchy, stats.expon, stats.laplace, stats.logistic, stats.norm]
# Tuple of `rtol` and `atol` for every tested moment
moments_tol = {1: (0., 2e-1), 2: (3e-1, 0.), 3: (4e-1, 8e-1), 4: (4., 0.)}


@pmp("distribution", dists)
def test_moment_consistency(distribution, plot=False):
    name = distribution.__name__.split('.')[-1]

    max_tree_depth = 20
    sampler = jft.NUTSChain(
        potential_energy=lambda x: -1 * distribution.logpdf(x),
        inverse_mass_matrix=1.,
        position_proto=jnp.array(0.),
        step_size=0.7193,
        max_tree_depth=max_tree_depth,
    )
    chain, _ = sampler.generate_n_samples(
        42, jnp.array(1.03890), num_samples=1000, save_intermediates=True
    )

    # unique, counts = jnp.unique(chain.depths, return_counts=True)
    # depths_frequencies = jnp.asarray((unique, counts)).T

    if plot is True:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2)

        bins = jnp.linspace(-10, 10)
        if distribution is stats.expon:
            bins = jnp.linspace(0, 10)
        axs.flat[0].hist(
            chain.samples, bins=bins, density=True, histtype="step"
        )
        axs.flat[0].plot(bins, distribution.pdf(bins), color='r')
        axs.flat[0].set_title(f"{name} PDF")

        axs.flat[1].hist(
            chain.depths,
            bins=jnp.arange(max_tree_depth + 1),
            density=True,
            histtype="step"
        )
        axs.flat[1].set_title(f"Tree-depth")
        fig.tight_layout()
        plt.show()

    # central moments; except for the first (i.e. mean)
    sample_moms_central = scipy.stats.moment(chain.samples, [1, 2, 3, 4, 5, 6])
    sample_moms_central[0] = jnp.mean(chain.samples)

    scipy_dist = getattr(scipy.stats, name)
    dist_moms_non_central = jnp.array(
        [scipy_dist.moment(i) for i in [1, 2, 3, 4, 5, 6]]
    )
    dist_moms_central = mnc2mc(dist_moms_non_central, wmean=True)

    for i, (smpl_mom, dist_mom) in enumerate(
        zip(sample_moms_central, dist_moms_central), start=1
    ):
        msg = (
            f"{name} (moment {i}) :: sampled: {smpl_mom:+.2e}"
            f" true: {dist_mom:+.2e} tested: "
        )
        print(msg, end="", file=sys.stderr)
        test = not jnp.isnan(dist_mom)
        test &= not (jnp.allclose(dist_mom, 0.) and i > 1)
        if i in moments_tol and test:
            assert_allclose(
                dist_mom, smpl_mom,
                **dict(zip(("rtol", "atol"), moments_tol[i]))
            )
            print("✓", file=sys.stderr)
        else:
            print("✗", file=sys.stderr)


if __name__ == "__main__":
    for d in dists:
        test_moment_consistency(d, plot=True)
