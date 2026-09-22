import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nifty.re as jft

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


@pmp("Ndim", [1, 10])
@pmp("r_min", [1.0e-5, 0.1])
@pmp("r_max", [1.0, 1.0e3])
@pmp("variance", [1.0, (1.0, 0.5)])
@pmp("lengthscale", [1.0, (1.0, 0.5)])
@pmp("negloglogslope", [3.0, (1.0, 0.1)])
@pmp("kcutoff", ["auto", 10.0])
@pmp("Ninterp", [128, "auto"])
@pmp("mode", ["ICR", "graphgp"])
def test_initialisation(
    Ndim, r_min, r_max, variance, lengthscale, negloglogslope, kcutoff, Ninterp, mode
):
    kernel = jft.MaternCovarianceModel(
        Ndim=Ndim,
        r_min=r_min,
        r_max=r_max,
        variance=variance,
        lengthscale=lengthscale,
        negloglogslope=negloglogslope,
        kcutoff=kcutoff,
        Ninterp=Ninterp,
        mode=mode,
    )
    assert isinstance(kernel, jft.Model)
    input = jft.random_like(jax.random.key(1), kernel.domain)
    res = kernel(input)
    if mode == "ICR":
        assert callable(res)
        cov = res(0.5, 0.0)
        assert jnp.all(jnp.isfinite(cov))
    elif mode == "graphgp":
        rs, cov = res
        assert isinstance(rs, jnp.ndarray)
        assert rs.ndim == 1
        assert jnp.all(jnp.isfinite(rs))
        assert isinstance(cov, jnp.ndarray)
        assert cov.ndim == 1
        assert jnp.all(jnp.isfinite(cov))
