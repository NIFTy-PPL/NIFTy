import sys

import pytest

pytest.importorskip("jax")

import jax
from jax import numpy as jnp
from numpy import ndarray

import nifty8.re as jft

NDARRAY_TYPE = [ndarray]

try:
    from jax.numpy import ndarray as jndarray

    NDARRAY_TYPE.append(jndarray)
except ImportError:
    pass

NDARRAY_TYPE = tuple(NDARRAY_TYPE)


def _json_serialize(obj):
    if isinstance(obj, NDARRAY_TYPE):
        return obj.tolist()
    raise TypeError(f"unknown type {type(obj)}")


def hashit(obj, n_chars=8) -> str:
    """Get first `n_chars` characters of Blake2B hash of `obj`."""
    import hashlib
    import json

    return hashlib.blake2b(
        bytes(json.dumps(obj, default=_json_serialize), "utf-8")
    ).hexdigest()[:n_chars]


def test_hmc_hash():
    """Test sapmler output against known hash from previous commits."""
    x0 = jnp.array([0.1, 1.223], dtype=jnp.float32)
    sampler = jft.HMCChain(
        potential_energy=lambda x: jnp.sum(x**2),
        inverse_mass_matrix=1.,
        position_proto=x0,
        step_size=0.193,
        num_steps=100,
        max_energy_difference=1.
    )
    chain, (key, pos) = sampler.generate_n_samples(
        key=42, initial_position=x0, num_samples=1000, save_intermediates=True
    )
    assert chain.divergences.sum() == 0
    accepted = chain.trees.accepted
    results = (pos, key, chain.samples, accepted)
    results_hash = hashit(results, n_chars=20)
    print(f"full hash: {results_hash}", file=sys.stderr)
    old_hash = "3d665689f809a98c81b3"
    assert results_hash == old_hash


def test_nuts_hash():
    """Test sapmler output against known hash from previous commits."""
    jax.config.update("jax_enable_x64", False)

    x0 = jnp.array([0.1, 1.223], dtype=jnp.float32)
    sampler = jft.NUTSChain(
        potential_energy=lambda x: jnp.sum(x**2),
        inverse_mass_matrix=1.,
        position_proto=x0,
        step_size=0.193,
        max_tree_depth=10,
        bias_transition=False,
        max_energy_difference=1.
    )
    chain, (key, pos) = sampler.generate_n_samples(
        key=42, initial_position=x0, num_samples=1000, save_intermediates=False
    )
    assert chain.divergences.sum() == 0
    results = (pos, key, chain.samples)
    results_hash = hashit(results, n_chars=20)
    print(f"full hash: {results_hash}", file=sys.stderr)
    old_hash = "8043850d7249acb77b26"
    assert results_hash == old_hash

    jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    test_hmc_hash()
    test_nuts_hash()
