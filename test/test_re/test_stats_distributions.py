import numpy as np
from numpy.testing import assert_allclose
import pytest

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
