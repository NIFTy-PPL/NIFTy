import pytest

pytest.importorskip("jax")

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

import nifty8.re as jft

pmp = pytest.mark.parametrize


@pmp("a", (3., 1.5, 4.))
@pmp("scale", (2., 4.))
@pmp("loc", (2., 4., 0.))
@pmp("seed", (42, 43))
def test_invgamma_roundtrip(a, scale, loc, seed, step=1e-1):
    rng = np.random.default_rng(seed)

    n_samples = int(1e+4)
    n_rvs = rng.normal(loc=0., scale=2., size=(n_samples, ))
    n_rvs = n_rvs.clip(-5.2, 5.2)

    pr = jft.invgamma_prior(a, scale, loc=loc, step=step)
    ipr = jft.invgamma_invprior(a, scale, loc=loc, step=step)

    n_roundtrip = ipr(pr(n_rvs))
    assert_allclose(n_roundtrip, n_rvs, rtol=1e-4, atol=1e-3)


@pmp("mean", (2., 4.))
@pmp("std", (2., 4.))
@pmp("seed", (42, 43))
def test_lognormal_roundtrip(mean, std, seed):
    rng = np.random.default_rng(seed)

    n_samples = int(1e+4)
    n_rvs = rng.normal(loc=0., scale=2., size=(n_samples, ))

    pr = jft.lognormal_prior(mean, std)
    ipr = jft.lognormal_invprior(mean, std)

    n_roundtrip = ipr(pr(n_rvs))
    assert_allclose(n_roundtrip, n_rvs, rtol=1e-6, atol=1e-6)


@pmp(
    "stats_and_prior", (
        (stats.norm(), jft.normal_prior(mean=0, std=1)),
        (stats.laplace(), jft.laplace_prior(alpha=1)),
        (
            stats.lognorm(s=1),
            jft.lognormal_prior(None, None, _log_mean=0, _log_std=1)
        ),
        (stats.invgamma(a=2), jft.invgamma_prior(a=2, scale=1)),
        (stats.uniform(), jft.uniform_prior(a_min=0, a_max=1)),
        (stats.norm(), jft.NormalPrior(mean=0, std=1)),
        (stats.laplace(), jft.LaplacePrior(alpha=1)),
        (
            stats.lognorm(s=1),
            jft.LogNormalPrior(
                np.exp(0.5),
                np.exp(0.5) * np.sqrt(np.exp(1) - 1)
            )
        ),
        (stats.invgamma(a=2), jft.InvGammaPrior(a=2, scale=1)),
        (stats.uniform(), jft.UniformPrior(a_min=0, a_max=1)),
    )
)
def test_quantiles(stats_and_prior):
    stats_distr, prior_dist = stats_and_prior
    atol = 0.0
    rtol = 1e-9 if not stats_distr.dist.name == "invgamma" else 1e-5

    q = np.linspace(1e-6, 1 - 1e-6, num=100, endpoint=True)
    pp = stats.norm.ppf(q, loc=0., scale=1.)
    assert_allclose(prior_dist(pp), stats_distr.ppf(q), rtol=rtol, atol=atol)
