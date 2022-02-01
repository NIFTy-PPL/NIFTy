from functools import partial
import sys

from jax import numpy as jnp
from jax import random
from jax import jit
from jax.config import config
import matplotlib.pyplot as plt

import nifty8.re as jft

config.update("jax_enable_x64", True)


@jit
def cosine_similarity(x, y):
    return jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))


def hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes)
    return tmp.real + tmp.imag


seed = 42
key = random.PRNGKey(seed)

dims = (1024, )

n_mgvi_iterations = 3
n_samples = 4
n_newton_iterations = 5
absdelta = 1e-4 * jnp.prod(jnp.array(dims))

cf = {"loglogavgslope": 2.}
loglogslope = cf["loglogavgslope"]
power_spectrum = lambda k: 1. / (k**loglogslope + 1.)

modes = jnp.arange((dims[0] / 2) + 1., dtype=float)
harmonic_power = power_spectrum(modes)
# Every mode appears exactly two times, first ascending then descending
# Save a little on the computational side by mirroring the ascending part
harmonic_power = jnp.concatenate((harmonic_power, harmonic_power[-2:0:-1]))

# Specify the model
correlated_field = lambda x: hartley(harmonic_power * x.val)
signal_response = lambda x: jnp.exp(1. + correlated_field(x))
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.Field(random.normal(shape=dims, key=key))
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = jnp.sqrt(noise_cov(jnp.ones(dims))
                      ) * random.normal(shape=dims, key=key)
data = signal_response_truth + noise_truth

nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
ham = jft.StandardHamiltonian(likelihood=nll).jit()

key, subkey = random.split(key)
pos_init = random.normal(shape=dims, key=subkey)
pos = 1e-2 * jft.Field(pos_init)

ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name")
)

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.}
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=pos,
        method="trust-ncg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=samples),
            "hessp": partial(ham_metric, primals_samples=samples),
            "initial_trust_radius": 1e+1,
            "max_trust_radius": 1e+4,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations,
            "name": "N",
            "subproblem_kwargs": {
                "miniter": 6,
            }
        }
    )
    # opt_state = jft.minimize(
    #     None,
    #     x0=pos,
    #     method="newton-cg",
    #     options={
    #         "fun_and_grad": partial(ham_vg, primals_samples=samples),
    #         "hessp": partial(ham_metric, primals_samples=samples),
    #         "absdelta": absdelta,
    #         "maxiter": n_newton_iterations
    #     }
    # )
    pos = opt_state.x
    print(
        (
            f"Post MGVI Iteration {i}: Energy {samples.at(pos).mean(ham)[0]:2.4e}"
            f"; Cos-Sim {cosine_similarity(pos.val, pos_truth.val):2.3%}"
            f"; #NaNs {jnp.isnan(pos.val).sum()}"
        ),
        file=sys.stderr
    )

post_sr_mean = jft.mean(tuple(signal_response(s) for s in samples.at(pos)))
fig, ax = plt.subplots()
ax.plot(signal_response_truth, alpha=0.7, label="Signal")
ax.plot(noise_truth, alpha=0.7, label="Noise")
ax.plot(data, alpha=0.7, label="Data")
ax.plot(post_sr_mean, alpha=0.7, label="Reconstruction")
ax.legend()
fig.tight_layout()
fig.savefig("cf_w_known_spectrum.png", dpi=400)
plt.close()
