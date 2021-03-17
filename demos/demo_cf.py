from jax.config import config
config.update("jax_enable_x64", True)

import sys
from typing import Union, Callable
from collections.abc import Iterable

from jax import numpy as np
from jax import random
from jax import jvp, vjp, value_and_grad, jit
from jax.ops import index_update

import jifty1 as jft


@jit
def hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes)
    return tmp.real + tmp.imag


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 42
    key = random.PRNGKey(seed)

    dims = (1024, )

    n_mgvi_iterations = 3
    n_samples = 4
    mirror_samples = True
    n_newton_iterations = 5

    cf_kw = {
        "zeromode": (1e-3, 1e-4),
        "fluctuations": (1e-1, 5e-3),
        "loglogavgslope": (-1., 0.5),
        "harmonic_domain_type": "Fourier"
    }
    amp = jft.Amplitude(dims, prefix="", **cf_kw)

    # Specify the model
    correlated_field = lambda x: hartley(amp(x))
    signal_response = lambda x: np.exp(correlated_field(x))
    noise_cov = lambda x: 0.1**2 * x
    noise_cov_inv = lambda x: 0.1**-2 * x

    # Create synthetic data
    key, subkey = random.split(key)
    pos_truth = jft.random_with_tree_shape(amp.tree_shape, key=subkey)
    signal_response_truth = signal_response(pos_truth)
    key, subkey = random.split(key)
    noise_truth = np.sqrt(noise_cov(np.ones(dims))
                         ) * random.normal(shape=dims, key=key)
    data = signal_response_truth + noise_truth

    nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
    ham = jft.StandardHamiltonian(likelihood=nll)

    key, subkey = random.split(key)
    pos_init = jft.random_with_tree_shape(amp.tree_shape, key=subkey)
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

        energy_vg = jit(value_and_grad(energy))
        met = jit(met)

        pos = jft.NCG(pos, energy_vg, met, n_newton_iterations)
        print(
            f"Post MGVI Iteration {i}: Energy {energy(pos):2.4e}",
            file=sys.stderr
        )

    fig, ax = plt.subplots()
    ax.plot(signal_response_truth, alpha=0.7, label="Signal")
    ax.plot(noise_truth, alpha=0.7, label="Noise")
    ax.plot(data, alpha=0.7, label="Data")
    ax.plot(signal_response(pos), alpha=0.7, label="Reconstruction")
    ax.legend()
    fig.tight_layout()
    plt.show()
