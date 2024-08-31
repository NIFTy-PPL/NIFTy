# f = exp( F * A * (
#       io(k, l0) +
#       slope(k)*(l-l0) +  # (-3 + deviations) Gauss(-3, 0.1)
#       Wiener(k, l-l0) (3d) - Intdl(wiener) (2d) - wiener(lmax-lmin)/(lmax-lmin) (2d)
# ) * (offset_mean + devations) )
# A(k=0)*(...) = (offset + dev)  -3 + Gaussian(mean + std)


import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from sys import exit

import nifty8.re as jft

shape = (128, )*2
distances = 0.1
freqs, reference_frequency = jnp.array((0.1, 1.5, 2, 10)), 1

zero_mode_settings = (-3.0, 0.1)
amplitude_settings = dict(
    scale=(0.4, 0.02),
    cutoff=(0.1, 0.01),
    loglogslope=(-4, 0.1),
)
amplitude_model = "matern"

# amplitude_settings = dict(
#     fluctuations=(1.0, 0.02),
#     loglogavgslope=(-4, 0.1),
#     flexibility=None,
#     asperity=None,
# )
# amplitude_model = "non_parametric"

spectral_amplitude_settings = dict(
    fluctuations=(1.0, 0.02),
    loglogavgslope=(-2., 0.1),
    flexibility=None,
    asperity=None,
)
# spectral_amplitude_settings = None
spectral_amplitude_model = "non_parametric"

spectral_idx_settings = dict(
    mean=(-1., .05),
    fluctuations=(.1, 1.e-2),
)
deviations_settings = dict(
    process='wiener',
    sigma=(0.2, 0.08),
)
# deviations_settings = None

mf_model = jft.build_default_mf_model(
    prefix='test',
    shape=shape,
    distances=distances,
    log_frequencies=freqs,
    reference_frequency_index=reference_frequency,
    zero_mode_settings=zero_mode_settings,
    spatial_amplitude_settings=amplitude_settings,
    spectral_index_settings=spectral_idx_settings,
    deviations_settings=deviations_settings,
    spatial_amplitude_model=amplitude_model,
    spectral_amplitude_settings=spectral_amplitude_settings,
    spectral_amplitude_model=spectral_amplitude_model
)


key = random.PRNGKey(42)

key = key + 1
figy, figx = 2, mf_model.target.shape[0] + 1
position = mf_model.init(key)
spatial_reference = mf_model.spatial_distribution(position)
spatial_mf_field = mf_model(position)

spectral_index = mf_model.spectral_index_distribution(position)
spectral_mf_field = mf_model.spectral_distribution(position)
spectral_deviations = mf_model.spectral_deviations_distribution(position)

if spectral_deviations is not None:
    figy = figy + 1

fig, axes = plt.subplots(figy, figx, sharex=True, sharey=True)
ax = axes[0]
im = ax[0].imshow(spatial_reference)
ax[0].set_title('spatial reference')
plt.colorbar(im, ax=ax[0])
for i, a in enumerate(ax[1:]):
    im = a.imshow(spatial_mf_field[i])
    plt.colorbar(im, ax=a)
    a.set_title(f'field nu={freqs[i]}')

ax = axes[1]
im = ax[0].imshow(spectral_index)
ax[0].set_title('spectral index')
plt.colorbar(im, ax=ax[0])
for i, a in enumerate(ax[1:]):
    im = a.imshow(spectral_mf_field[i])
    plt.colorbar(im, ax=a)
    a.set_title(f'spectral nu={freqs[i]}')

if spectral_deviations is not None:
    ax = axes[2]
    for i, a in enumerate(ax[1:]):
        im = a.imshow(spectral_deviations[i])
        plt.colorbar(im, ax=a)
        a.set_title(f'spectral devs nu={freqs[i]}')

plt.show()
