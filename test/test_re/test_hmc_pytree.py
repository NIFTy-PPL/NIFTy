import pytest

pytest.importorskip("jax")

from functools import partial

from jax import numpy as jnp
from jax.tree_util import tree_leaves
from numpy.testing import assert_array_equal

import nifty8.re as jft


def test_hmc_pytree():
    """Test sapmler output against known hash from previous commits."""
    initial_position = jnp.array([0.31415, 2.71828])

    sampler_init = partial(
        jft.HMCChain,
        potential_energy=lambda x: jft.vdot(x, x),
        inverse_mass_matrix=1.,
        step_size=0.193,
        num_steps=100
    )

    initial_position_py = jft.Vector(({"lvl0": initial_position}, ))
    smpl_w_pytree = sampler_init(position_proto=initial_position_py
                                ).generate_n_samples(
                                    key=321,
                                    initial_position=initial_position_py,
                                    num_samples=1000
                                )
    smpl_wo_pytree = sampler_init(position_proto=initial_position
                                 ).generate_n_samples(
                                     key=321,
                                     initial_position=initial_position,
                                     num_samples=1000
                                 )

    ts_w, ts_wo = tree_leaves(smpl_w_pytree), tree_leaves(smpl_wo_pytree)
    assert len(ts_w) == len(ts_wo)
    for w, wo in zip(ts_w, ts_wo):
        assert_array_equal(w, wo)


def test_nuts_pytree():
    """Test sapmler output against known hash from previous commits."""
    initial_position = jnp.array([0.31415, 2.71828])

    sampler_init = partial(
        jft.NUTSChain,
        potential_energy=lambda x: jft.vdot(x, x),
        inverse_mass_matrix=1.,
        step_size=0.193,
        max_tree_depth=10,
    )

    initial_position_py = jft.Vector(({"lvl0": initial_position}, ))
    smpl_w_pytree = sampler_init(position_proto=initial_position_py
                                ).generate_n_samples(
                                    key=323,
                                    initial_position=initial_position_py,
                                    num_samples=1000
                                )
    smpl_wo_pytree = sampler_init(position_proto=initial_position
                                 ).generate_n_samples(
                                     key=323,
                                     initial_position=initial_position,
                                     num_samples=1000
                                 )

    ts_w, ts_wo = tree_leaves(smpl_w_pytree), tree_leaves(smpl_wo_pytree)
    assert len(ts_w) == len(ts_wo)
    for w, wo in zip(ts_w, ts_wo):
        assert_array_equal(w, wo)


if __name__ == "__main__":
    test_hmc_pytree()
    test_nuts_pytree()
