from jax.config import config

config.update("jax_enable_x64", True)

import sys

from jax import numpy as np
from jax import random
from jax import value_and_grad, jit

import jifty1 as jft


@jit
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


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
    n_newton_iterations = 5
    absdelta = 1e-4 * np.prod(np.array(dims))

    cf = {"loglogavgslope": 2.}
    loglogslope = cf["loglogavgslope"]
    power_spectrum = lambda k: 1. / (k**loglogslope + 1.)

    modes = np.arange((dims[0] / 2) + 1., dtype=float)
    harmonic_power = power_spectrum(modes)
    # Every mode appears exactly two times, first ascending then descending
    # Save a little on the computational side by mirroring the ascending part
    harmonic_power = np.concatenate((harmonic_power, harmonic_power[-2:0:-1]))

    # Specify the model
    correlated_field = lambda x: hartley(harmonic_power * x.val)
    signal_response = lambda x: np.exp(1. + correlated_field(x))
    noise_cov = lambda x: 0.1**2 * x
    noise_cov_inv = lambda x: 0.1**-2 * x

    # Create synthetic data
    key, subkey = random.split(key)
    pos_truth = jft.Field(random.normal(shape=dims, key=key))
    signal_response_truth = signal_response(pos_truth)
    key, subkey = random.split(key)
    noise_truth = np.sqrt(noise_cov(np.ones(dims))
                         ) * random.normal(shape=dims, key=key)
    data = signal_response_truth + noise_truth

    nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
    ham = jft.StandardHamiltonian(likelihood=nll).jit()

    key, subkey = random.split(key)
    pos_init = random.normal(shape=dims, key=subkey)
    pos = 1e-2 * jft.Field(pos_init)

    kl_energy = jft.vmap_forest_mean(
        lambda p, s: ham.energy(p + s), in_axes=(None, 0)
    )
    kl_vag = jit(value_and_grad(kl_energy))
    kl_metric = jit(jft.vmap_forest_mean(
        lambda p, s, t: ham.metric(p + s, t),
        in_axes=(None, 0, None)
    ))

    # Minimize the potential
    for i in range(n_mgvi_iterations):
        print(f"MGVI Iteration {i}", file=sys.stderr)
        print("Sampling...", file=sys.stderr)
        key, subkey = random.split(key, 2)
        mkl = jft.MetricKL(
            ham,
            pos,
            n_samples=n_samples,
            key=subkey,
            mirror_samples=True,
            linear_sampling_kwargs={"absdelta": absdelta / 10.}
        )
        kl_vag_at_p = lambda p: kl_vag(p, tuple(mkl.samples))
        kl_metric_at = lambda p, t: kl_metric(p, tuple(mkl.samples), t)

        print("Minimizing...", file=sys.stderr)
        # TODO: Re-introduce a simplified version that works without fields
        opt_state = jft.minimize(
            None,
            x0=pos,
            method="newton-cg",
            options={
                "fun_and_grad": kl_vag_at_p,
                "hessp": kl_metric_at,
                "absdelta": absdelta,
                "maxiter": n_newton_iterations
            }
        )
        pos = opt_state.x
        print(
            (
                f"Post MGVI Iteration {i}: Energy {mkl(pos):2.4e}"
                f"; Cos-Sim {cosine_similarity(pos.val, pos_truth.val):2.3%}"
                f"; #NaNs {np.isnan(pos.val).sum()}"
            ),
            file=sys.stderr
        )

    post_sr_mean = jft.mean(
        tuple(signal_response(pos + s) for s in mkl.samples)
    )
    fig, ax = plt.subplots()
    ax.plot(signal_response_truth, alpha=0.7, label="Signal")
    ax.plot(noise_truth, alpha=0.7, label="Noise")
    ax.plot(data, alpha=0.7, label="Data")
    ax.plot(post_sr_mean, alpha=0.7, label="Reconstruction")
    ax.legend()
    fig.tight_layout()
    fig.savefig("cf_w_known_spectrum.png", dpi=400)
    plt.close()
