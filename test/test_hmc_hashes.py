import sys
from numpy import ndarray
from jax import numpy as np

from jifty1 import hmc

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
    sampler = hmc.HMCChain(
        potential_energy=lambda x: np.sum(x**2),
        inverse_mass_matrix=1.,
        initial_position=np.array([0.1, 1.223]),
        key=42,
        step_size=0.193,
        num_steps=100,
        dbg_info=True,
        compile=True,
        max_energy_difference=1.
    )
    chain = sampler.generate_n_samples(1000)
    assert chain.divergences.sum() == 0
    key, pos = sampler.last_state
    accepted = chain.trees.accepted
    results = (pos, key, chain.samples, accepted)
    results_hash = hashit(results, n_chars=20)
    print(f"full hash: {results_hash}", file=sys.stderr)
    old_hash = "a62bbc84432eb4f13ad6"
    assert results_hash == old_hash


def test_nuts_hash():
    """Test sapmler output against known hash from previous commits."""
    sampler = hmc.NUTSChain(
        potential_energy=lambda x: np.sum(x**2),
        inverse_mass_matrix=1.,
        initial_position=np.array([0.1, 1.223]),
        step_size=0.193,
        max_tree_depth=10,
        key=42,
        dbg_info=False,
        bias_transition=False,
        compile=True,
        max_energy_difference=1.
    )
    chain = sampler.generate_n_samples(1000)
    assert chain.divergences.sum() == 0
    key, pos = sampler.last_state
    results = (pos, key, chain.samples)
    results_hash = hashit(results, n_chars=20)
    print(f"full hash: {results_hash}", file=sys.stderr)
    old_hash = "ba324b77e6d73afae514"
    assert results_hash == old_hash


if __name__ == "__main__":
    test_hmc_hash()
    test_nuts_hash()
