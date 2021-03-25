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

    n_mgvi_iterations = 3
    n_samples = 4
    n_newton_iterations = 10

    dims_ax1 = (128, )
    dims_ax2 = (256, )
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
    d = 1. / dims_ax1[0]
    cfm.add_fluctuations(dims_ax1, distances=d, **cf_fl, prefix="ax1")
    d = 1. / dims_ax2[0]
    cfm.add_fluctuations(dims_ax2, distances=d, **cf_fl, prefix="ax2")
    correlated_field, ptree = cfm.finalize()

    signal_response = lambda x: correlated_field(x)
    noise_cov = lambda x: 0.1**2 * x
    noise_cov_inv = lambda x: 0.1**-2 * x

    # Create synthetic data
    key, subkey = random.split(key)
    pos_truth = jft.random_with_tree_shape(ptree, key=subkey)
    signal_response_truth = signal_response(pos_truth)
    key, subkey = random.split(key)
    noise_truth = np.sqrt(
        noise_cov(np.ones(signal_response_truth.shape))
    ) * random.normal(shape=signal_response_truth.shape, key=key)
    data = signal_response_truth + noise_truth

    nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
    ham = jft.StandardHamiltonian(likelihood=nll).jit()
    draw = lambda p, k: ham.draw_sample(p, key=k, from_inverse=True)
    ham_energy_vg = jit(value_and_grad(ham))

    key, subkey = random.split(key)
    pos_init = jft.random_with_tree_shape(ptree, key=subkey)
    pos = jft.Field(pos_init.copy())

    # Minimize the potential
    for i in range(n_mgvi_iterations):
        print(f"MGVI Iteration {i}", file=sys.stderr)
        key, *subkeys = random.split(key, 1 + n_samples)
        print("Sampling...", file=sys.stderr)
        samples = []
        samples = [draw(pos, k) for k in subkeys]
        samples += [-s for s in samples]

        def energy_vg(p):
            return jft.mean(tuple(ham_energy_vg(p + s) for s in samples))

        def met(p, t):
            return jft.mean(tuple(ham.metric(p + s, t) for s in samples))

        print("Minimizing...", file=sys.stderr)
        pos = jft.newton_cg(pos, energy_vg, met, n_newton_iterations)
        msg = f"Post MGVI Iteration {i}: Energy {energy_vg(pos)[0]:2.4e}"
        print(msg, file=sys.stderr)

    namps = cfm.get_normalized_amplitudes()
    post_sr_mean = jft.mean(tuple(signal_response(pos + s) for s in samples))
    post_namps1_mean = jft.mean(tuple(namps[0](pos + s)[1:] for s in samples))
    post_namps2_mean = jft.mean(tuple(namps[1](pos + s)[1:] for s in samples))
    to_plot = [
        ("Signal", signal_response_truth, "im"),
        ("Noise", noise_truth, "im"),
        ("Data", data, "im"),
        ("Reconstruction", post_sr_mean, "im"),
        ("Ax1", (namps[0](pos_truth)[1:], post_namps1_mean), "loglog"),
        ("Ax2", (namps[1](pos_truth)[1:], post_namps2_mean), "loglog"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (title, field, tp) in zip(axs.flat, to_plot):
        ax.set_title(title)
        if tp == "im":
            im = ax.imshow(field, cmap="inferno")
            plt.colorbar(im, ax=ax, orientation="horizontal")
        else:
            ax_plot = ax.loglog if tp == "loglog" else ax.plot
            field = field if isinstance(field, (tuple, list)) else (field, )
            for f in field:
                ax_plot(f, alpha=0.7)
    fig.tight_layout()
    plt.show()
