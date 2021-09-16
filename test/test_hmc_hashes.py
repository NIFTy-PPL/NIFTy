from jifty1 import hmc
import jax.numpy as np
import hashlib

def deterministic_hash(arg):
    return hashlib.md5(str(arg).encode('utf-8')).digest()

def test_hmc_hash():
    """Test sapmler output against known hash from previous commits."""
    sampler = hmc.HMCChain(
        initial_position = np.array([0.1, 1.223]),
        potential_energy = lambda x: np.sum(x**2),
        diag_mass_matrix = 1.,
        eps = 0.193,
        n_of_integration_steps = 100,
        rngseed = 42,
        dbg_info = True,
        compile = True 
    )
    results = sampler.generate_n_samples(1000)
    results_hash = deterministic_hash(str(results))
    print(f"{results_hash=}")
    assert results_hash == b"t{\x059'\x12\x85\x8do\xf6\x1a/\xb1\xaa!\xe0"

def test_nuts_hash():
    """Test sapmler output against known hash from previous commits."""
    sampler = hmc.NUTSChain(
        initial_position = np.array([0.1, 1.223]),
        potential_energy = lambda x: np.sum(x**2),
        diag_mass_matrix = 1.,
        eps = 0.193,
        maxdepth = 10,
        rngseed = 42,
        dbg_info = True,
        compile = True
    )
    results = sampler.generate_n_samples(1000)
    results_hash = deterministic_hash(str(results))
    print(f"{results_hash=}")
    assert results_hash == b'\x19\xc6\x9a\x17\x1f\xbc\xf9\x1d\x96?\xa0\x85v\x8cP-'

if __name__ == "__main__":
    test_hmc_hash()
    test_nuts_hash()