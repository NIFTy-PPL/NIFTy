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
        initial_position=np.array([0.1, 1.223]),
        potential_energy=lambda x: np.sum(x**2),
        diag_mass_matrix=1.,
        step_size=0.193,
        n_of_integration_steps=100,
        rngseed=42,
        dbg_info=False,
        compile=True
    )
    results = sampler.generate_n_samples(1000)
    results_hash = hashit(results, n_chars=20)
    print(f"full hash: {results_hash}", file=sys.stderr)
    old_hash = "dd15f689f20d16ff1480"
    assert results_hash == old_hash


def test_nuts_hash():
    """Test sapmler output against known hash from previous commits."""
    sampler = hmc.NUTSChain(
        initial_position=np.array([0.1, 1.223]),
        potential_energy=lambda x: np.sum(x**2),
        diag_mass_matrix=1.,
        step_size=0.193,
        maxdepth=10,
        rngseed=42,
        dbg_info=False,
        bias_transition=False,
        compile=True
    )
    results = sampler.generate_n_samples(1000)
    results_hash = hashit(results, n_chars=20)
    print(f"full hash: {results_hash}", file=sys.stderr)
    old_hash = "ba324b77e6d73afae514"
    assert results_hash == old_hash


if __name__ == "__main__":
    test_hmc_hash()
    test_nuts_hash()
