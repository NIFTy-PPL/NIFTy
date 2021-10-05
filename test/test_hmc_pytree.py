import sys
import numpy as np
from functools import partial
from jax import numpy as jnp
from jax.tree_util import tree_leaves

import jifty1 as jft
from jifty1 import hmc


def test_hmc_pytree():
    """Test sapmler output against known hash from previous commits."""
    initial_position = jnp.array([0.31415, 2.71828])

    sampler_init = partial(
        hmc.HMCChain,
        potential_energy=jft.sum_of_squares,
        inverse_mass_matrix=1.,
        step_size=0.193,
        num_steps=100,
        key=321,
        dbg_info=False,
        compile=True
    )

    smpl_w_pytree = sampler_init(
        initial_position=jft.Field(({
            "lvl0": initial_position
        }, ))
    ).generate_n_samples(1000)
    smpl_wo_pytree = sampler_init(initial_position=initial_position
                                ).generate_n_samples(1000)

    ts_w, ts_wo = tree_leaves(smpl_w_pytree), tree_leaves(smpl_wo_pytree)
    assert len(ts_w) == len(ts_wo)
    for w, wo in zip(ts_w, ts_wo):
        np.testing.assert_array_equal(w, wo)


def test_nuts_pytree():
    """Test sapmler output against known hash from previous commits."""
    initial_position = jnp.array([0.31415, 2.71828])

    sampler_init = partial(
        hmc.NUTSChain,
        potential_energy=jft.sum_of_squares,
        inverse_mass_matrix=1.,
        step_size=0.193,
        max_tree_depth=10,
        key=323,
        dbg_info=False,
        compile=True
    )

    smpl_w_pytree = sampler_init(
        initial_position=jft.Field(({
            "lvl0": initial_position
        }, ))
    ).generate_n_samples(1000)
    smpl_wo_pytree = sampler_init(initial_position=initial_position
                                ).generate_n_samples(1000)

    ts_w, ts_wo = tree_leaves(smpl_w_pytree), tree_leaves(smpl_wo_pytree)
    assert len(ts_w) == len(ts_wo)
    for w, wo in zip(ts_w, ts_wo):
        np.testing.assert_array_equal(w, wo)


if __name__ == "__main__":
    test_hmc_pytree()
    test_nuts_pytree()
