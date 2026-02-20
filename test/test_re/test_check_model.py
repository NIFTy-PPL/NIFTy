import jax
import jax.numpy as jnp
import nifty.re as jft


jax.config.update("jax_enable_x64", True)


def test_check_model():
    a = jnp.full((10,), 3.5)

    def func(x):
        return a * x

    jft.check_model(func, a)

    model = jft.Model(call=func, domain=jax.ShapeDtypeStruct(a.shape, a.dtype))
    jft.check_model(model, a)
