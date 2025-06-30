import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import nifty.re as jft

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


@pmp("eval_dtype", ["float32", "float64"])
@pmp(
    "name, stats_distr, prior_dist",
    # wrap `prior_dist` in a lambda to defer instantiation to after `jax.config.enable_x64` is adjusted
    # if we do not do this, tests for float32 will internally still use float64
    [
        ("normal_prior call", stats.norm(), lambda: jft.normal_prior(mean=0, std=1)),
        ("laplace_prior call", stats.laplace(), lambda: jft.laplace_prior(alpha=1)),
        (
            "lognormal_prior call",
            stats.lognorm(s=1),
            lambda: jft.lognormal_prior(None, None, _log_mean=0, _log_std=1),
        ),
        (
            "invgamma_prior call",
            stats.invgamma(a=2),
            lambda: jft.invgamma_prior(a=2, scale=1),
        ),
        (
            "uniform_prior call",
            stats.uniform(),
            lambda: jft.uniform_prior(a_min=0, a_max=1),
        ),
        ("NormalPrior model", stats.norm(), lambda: jft.NormalPrior(mean=0, std=1)),
        ("LaplacePrior model", stats.laplace(), lambda: jft.LaplacePrior(alpha=1)),
        (
            "LogNormalPrior model",
            stats.lognorm(s=1),
            lambda: jft.LogNormalPrior(
                np.exp(0.5), np.exp(0.5) * np.sqrt(np.exp(1) - 1)
            ),
        ),
        (
            "InvGammaPrior model",
            stats.invgamma(a=2),
            lambda: jft.InvGammaPrior(a=2, scale=1),
        ),
        (
            "UniformPrior model",
            stats.uniform(),
            lambda: jft.UniformPrior(a_min=0, a_max=1),
        ),
    ],
)
def test_quantiles(name, stats_distr, prior_dist, eval_dtype):
    # test for latent values from -8.2 to 8.2
    pp = np.linspace(-8.2, 8.2, num=100, endpoint=True)
    q = stats.norm.cdf(pp, loc=0.0, scale=1.0)

    # test in float32-only mode when eval_dtype == 'float32'
    jax.config.update("jax_enable_x64", True if eval_dtype == "float64" else False)
    pp = jax.numpy.array(pp, dtype=eval_dtype)
    prior_dist = prior_dist()  # instantiate prior fn

    # evaluate transfer functions
    gt = stats_distr.ppf(q)
    ours = prior_dist(pp)

    # catch NaNs
    assert not np.isnan(ours).any()

    # set tolerances based on distributions and eval_dtypes
    if stats_distr.dist.name == "invgamma":
        # InverseGamma prior uses interpolation, error independent of dtype
        rtol = np.full_like(gt, 1e-5)
        rtol[pp >= 7] = 1e-2  # higher error for pp values >= 7
    elif stats_distr.dist.name == "uniform":
        rtol = 1e-5 if eval_dtype == "float32" else 1e-12
        rtol = np.full_like(gt, rtol)
    else:
        rtol = 1e-6 if eval_dtype == "float32" else 1e-9
        rtol = np.full_like(gt, rtol)
        # scipy.norm.cdf becomes imprecise for high pp values
        # dynamically adjust tolerance
        rtol_dyn = 10 ** (-1.5 - 2 * (8.2 - pp))
        rtol = np.max([rtol, rtol_dyn], axis=0)

    # allclose cannot handle per-element tolerance specification
    # slice arrays by tolerance level of entries
    for rtol_test in np.unique(rtol):
        idx = rtol == rtol_test
        assert_allclose(ours[idx], gt[idx], rtol=rtol_test, atol=0.0)
