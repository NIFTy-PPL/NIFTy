import sys
from jax import numpy as np

from jifty1 import hmc


def hashit(obj, n_chars=8) -> str:
    """Get first `n_chars` characters of Blake2B hash of `obj`."""
    import hashlib

    return hashlib.blake2b(bytes(str(obj), "utf-8")).hexdigest()[:n_chars]


def test_hmc_hash():
    """Test sapmler output against known hash from previous commits."""
    sampler = hmc.HMCChain(
        initial_position=np.array([0.1, 1.223]),
        potential_energy=lambda x: np.sum(x**2),
        diag_mass_matrix=1.,
        eps=0.193,
        n_of_integration_steps=100,
        rngseed=42,
        dbg_info=False,
        compile=True
    )
    results = sampler.generate_n_samples(1000)
    results_hash = hashit(results, n_chars=20)
    print(f"full hash: {results_hash}", file=sys.stderr)
    old_hash = "c44e34125a942c54c785"
    assert results_hash == old_hash


def test_nuts_hash():
    """Test sapmler output against known hash from previous commits."""
    sampler = hmc.NUTSChain(
        initial_position=np.array([0.1, 1.223]),
        potential_energy=lambda x: np.sum(x**2),
        diag_mass_matrix=1.,
        eps=0.193,
        maxdepth=10,
        rngseed=42,
        dbg_info=False,
        compile=True
    )
    results = sampler.generate_n_samples(1000)
    results_hash = hashit(results, n_chars=20)
    print(f"full hash: {results_hash}", file=sys.stderr)
    old_hash = "de1f73aa40f3873d8023"
    assert results_hash == old_hash


if __name__ == "__main__":
    test_hmc_hash()
    test_nuts_hash()
