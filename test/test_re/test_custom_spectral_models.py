# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from numpy.testing import assert_allclose

def test_custom_amplitude_registration():
    maker = jft.CorrelatedFieldMaker("test_")
    maker.set_amplitude_total_offset(0.0, (1.0, 0.1))
    created = []

    def factory(grid, prefix):
        assert prefix == "test_time_"
        assert grid.shape == (32,)
        assert_allclose(grid.distances, (0.5,))
        name = prefix + "scale"

        def amplitude(p):
            # The maker overrides the zero mode of this unnormalized spectrum.
            return jnp.full(grid.harmonic_grid.mode_lengths.shape,
                            3.0 * jnp.exp(p[name]))

        model = jft.Model(
            amplitude, domain={name: jft.ShapeWithDtype(())}, white_init=True
        )
        created.append(model)
        return model

    maker.add_fluctuations_custom((32,), 0.5, factory, prefix="time_")
    assert len(maker.fluctuations) == len(created) == 1
    parameters = {"test_time_scale": jnp.array(0.0)}
    assert_allclose(created[0](parameters)[0], 3.0)
    amplitude = maker.fluctuations[0](parameters)
    assert_allclose(amplitude[0], 16.0)
    assert_allclose(amplitude[1:], 3.0)

    field = maker.finalize()
    assert set(field.domain) == {"test_time_scale", "test_zeromode", "test_xi"}
    assert field.target.shape == (32,)
    parameters = field.init(jax.random.PRNGKey(42))
    output = jax.jit(field)(parameters)
    assert np.all(np.isfinite(output))
    gradient = jax.grad(lambda p: jnp.sum(field(p) ** 2))(parameters)
    assert np.isfinite(gradient["test_time_scale"])
    assert abs(gradient["test_time_scale"]) > 0


def test_custom_factory_validation():
    maker = jft.CorrelatedFieldMaker('test_')
    with pytest.raises(TypeError, match='return a Model'):
        maker.add_fluctuations_custom(16, 1.0, lambda grid, prefix: None)
    with pytest.raises(
        ValueError,
        match=r'must return one value per unique harmonic mode length; '
              r'expected output shape \(9,\), got \(2,\)',
    ):
        maker.add_fluctuations_custom(
            16, 1.0, lambda grid, prefix: jft.Model(
                lambda p: jnp.ones((2,)), domain={}, white_init=True
            )
        )
    assert maker.fluctuations == ()
