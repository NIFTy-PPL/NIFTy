import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


@pmp("a", (3.0, 1.5, 4.0))
@pmp("scale", (2.0, 4.0))
@pmp("loc", (2.0, 4.0, 0.0))
@pmp("seed", (42, 43))
def test_invgamma_roundtrip(a, scale, loc, seed, step=1e-1):
    rng = np.random.default_rng(seed)

    n_samples = int(1e4)
    n_rvs = rng.normal(loc=0.0, scale=2.0, size=(n_samples,))
    n_rvs = n_rvs.clip(-5.2, 5.2)

    pr = jft.invgamma_prior(a, scale, loc=loc, step=step)
    ipr = jft.invgamma_invprior(a, scale, loc=loc, step=step)

    n_roundtrip = ipr(pr(n_rvs))
    assert_allclose(n_roundtrip, n_rvs, rtol=1e-4, atol=1e-3)


@pmp("mean", (2.0, 4.0))
@pmp("std", (2.0, 4.0))
@pmp("seed", (42, 43))
def test_lognormal_roundtrip(mean, std, seed):
    rng = np.random.default_rng(seed)

    n_samples = int(1e4)
    n_rvs = rng.normal(loc=0.0, scale=2.0, size=(n_samples,))

    pr = jft.lognormal_prior(mean, std)
    ipr = jft.lognormal_invprior(mean, std)

    n_roundtrip = ipr(pr(n_rvs))
    assert_allclose(n_roundtrip, n_rvs, rtol=1e-6, atol=1e-6)


@pmp("eval_dtype", ['float32', 'float64'])
@pmp(
    "name, stats_distr, prior_dist",
    [
        ('normal_prior call', stats.norm(), jft.normal_prior(mean=0, std=1)),
        ('laplace_prior call', stats.laplace(), jft.laplace_prior(alpha=1)),
        ('lognormal_prior call', stats.lognorm(s=1), jft.lognormal_prior(None, None, _log_mean=0, _log_std=1)),
        ('invgamma_prior call', stats.invgamma(a=2), jft.invgamma_prior(a=2, scale=1)),
        ('uniform_prior call', stats.uniform(), jft.uniform_prior(a_min=0, a_max=1)),
        ('NormalPrior model', stats.norm(), jft.NormalPrior(mean=0, std=1)),
        ('LaplacePrior model', stats.laplace(), jft.LaplacePrior(alpha=1)),
        (
            'LogNormalPrior model',
            stats.lognorm(s=1),
            jft.LogNormalPrior(np.exp(0.5), np.exp(0.5) * np.sqrt(np.exp(1) - 1)),
        ),
        ('InvGammaPrior model', stats.invgamma(a=2), jft.InvGammaPrior(a=2, scale=1)),
        ('UniformPrior model', stats.uniform(), jft.UniformPrior(a_min=0, a_max=1)),
    ],
)
def test_quantiles(name, stats_distr, prior_dist, eval_dtype):
    pp = np.linspace(-8.2, 8.2, num=100, endpoint=True)
    q = stats.norm.cdf(pp, loc=0.0, scale=1.0)

    pp = jax.numpy.array(pp, dtype=eval_dtype)
    q = jax.numpy.array(q, dtype=eval_dtype)

    gt = stats_distr.ppf(q)
    ours = prior_dist(pp)

    atol = 0.0
    rtol = 1e-9 if not stats_distr.dist.name == "invgamma" else 1e-5

    # adapt tolerance level for high pp values to account for scipy.norm.cdf becoming somewhat inaccurate
    rtol = np.full_like(pp, rtol)
    for i in (5.67, 6, 6.33, 6.67, 7, 7.33, 7.67, 8):
        rtol[pp > i] *= 10

    assert not np.any(np.isnan(ours))

    # allclose cannot handle per-element tolerance specification
    # slice arrays by tolerance level of entries
    for rtol_test in np.unique(rtol):
        idx = (rtol == rtol_test)
        assert_allclose(ours[idx], gt[idx], rtol=rtol_test, atol=atol)
