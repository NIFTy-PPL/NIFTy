import jax
import nifty8.re as jft
import pytest

jax.config.update("jax_enable_x64", True)


def test_map_forest_axes_validation():
    f = lambda x: x
    jft.map_forest(f, in_axes=1)
    with pytest.raises(ValueError):
        jft.map_forest(f, in_axes=(None,))
    with pytest.raises(TypeError):
        jft.map_forest(f, in_axes=(1.0,))
