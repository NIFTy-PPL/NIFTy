# f = exp( F * A * (
#       io(k, l0) +
#       slope(k)*(l-l0) +  # (-3 + deviations) Gauss(-3, 0.1)
#       Wiener(k, l-l0) (3d) - Intdl(wiener) (2d) - wiener(lmax-lmin)/(lmax-lmin) (2d)
# ) * (offset_mean + devations) )
# A(k=0)*(...) = (offset + dev)  -3 + Gaussian(mean + std)


import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# from src.re.library.mf_model import build_mf_model
from src.re import build_mf_model


shape, distances = (128,)*2, (0.1,)*2
freqs, reference_frequency = jnp.array((0.1, 1.5, 2, 10)), 1

zero_mode_settings = dict(
    mean=-3,
    deviations=(0.0, 0.1)
)
amplitude_settings = dict(
    scale=(1.0, 0.02),
    cutoff=(0.1, 0.01),
    loglogslope=(-4, 0.1),
)
slope_settings = dict(
    mean=(-4.3, .05),
)
deviations_settings = dict(
    process='wiener',
    sigma=(0.1, 0.01),
)

mf_model = build_mf_model(
    prefix='test',
    shape_2d=shape,
    distances_2d=distances,
    log_frequencies=freqs,
    reference_frequency=reference_frequency,

    zero_mode_settings=zero_mode_settings,
    amplitude_settings=amplitude_settings,
    slope_settings=slope_settings,
    deviations_settings=deviations_settings,
)

key = random.PRNGKey(42)

key = key + 1
fi = mf_model(mf_model.init(key))
fig, axes = plt.subplots(1, len(fi), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    im = ax.imshow(fi[i])
    plt.colorbar(im, ax=ax)
plt.show()
