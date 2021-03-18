from jax.config import config
config.update("jax_enable_x64", True)

import sys
from typing import Union, Callable
from collections.abc import Iterable

from jax import numpy as np
from jax import random
from jax import jvp, vjp, value_and_grad, jit

import jifty1 as jft


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 42
    key = random.PRNGKey(seed)

    dims = (1024, )

    n_mgvi_iterations = 3
    n_samples = 4
    mirror_samples = True
    n_newton_iterations = 10

    cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
    cf_fl = {
        "fluctuations": (1e-1, 5e-3),
        "loglogavgslope": (-1., 1e-2),
        "flexibility": (1e+0, 5e-1),
        "asperity": (5e-1, 1e-1),
        "harmonic_domain_type": "Fourier"
    }
    cfm = jft.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(**cf_zm)
    cfm.add_fluctuations(dims, **cf_fl, prefix="")
    correlated_field, ptree = cfm.finalize()

    signal_response = lambda x: np.exp(correlated_field(x))
    noise_cov = lambda x: 0.1**2 * x
    noise_cov_inv = lambda x: 0.1**-2 * x

    # Create synthetic data
    key, subkey = random.split(key)
    pos_truth = jft.random_with_tree_shape(ptree, key=subkey)
    signal_response_truth = signal_response(pos_truth)
    key, subkey = random.split(key)
    noise_truth = np.sqrt(noise_cov(np.ones(dims))
                         ) * random.normal(shape=dims, key=key)
    data = signal_response_truth + noise_truth

    nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
    ham = jft.StandardHamiltonian(likelihood=nll).jit()

    key, subkey = random.split(key)
    pos_init = jft.random_with_tree_shape(ptree, key=subkey)
    pos = pos_init.copy()

    # Minimize the potential
    for i in range(n_mgvi_iterations):
        print(f"MGVI Iteration {i}", file=sys.stderr)
        key, *subkeys = random.split(key, 1 + n_samples)
        samples = []
        draw = lambda k: ham.draw_sample(pos, key=k, from_inverse=True)[0]
        samples = [jft.makeField(draw(k)) for k in subkeys]
        samples += [-s for s in samples]

        def energy(p):
            p = jft.makeField(p)
            rdc = sum(ham((p + s).to_tree()) for s in samples)
            return 1. / len(samples) * rdc

        def met(p, t):
            p = jft.makeField(p)
            rdc = sum(
                jft.makeField(ham.metric((p + s).to_tree(), t)) for s in samples
            )
            return (1. / len(samples) * rdc).to_tree()

        energy_vg = value_and_grad(energy)
        met = met

        pos = jft.NCG(pos, energy_vg, met, n_newton_iterations)
        print(
            f"Post MGVI Iteration {i}: Energy {energy(pos):2.4e}",
            file=sys.stderr
        )

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(signal_response_truth, alpha=0.7, label="Signal")
    axs[0].plot(noise_truth, alpha=0.7, label="Noise")
    axs[0].plot(data, alpha=0.7, label="Data")
    axs[0].plot(signal_response(pos), alpha=0.7, label="Reconstruction")
    axs[0].legend()
    axs[1].loglog(cfm.get_amplitude()(pos_truth)[1:], alpha=0.7, label="Signal")
    axs[1].loglog(cfm.get_amplitude()(pos)[1:], alpha=0.7, label="Reconstruction")
    fig.tight_layout()
    plt.show()
