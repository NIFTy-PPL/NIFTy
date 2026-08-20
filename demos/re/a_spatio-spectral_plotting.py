#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# # Spatio-spectral false-colour plotting
#
# Renders a 3-D (two spatial, one spectral) correlated field as a single sRGB
# image, with hue encoding spectral shape and brightness encoding flux.
#
# This demo is about getting a *realistic* field to look at, and about the one
# knob that matters most for it: linear versus logarithmic luminance mapping.
# For the projector interface itself — flux conventions, ways of choosing the
# luminance range, highlight handling, legends — see the "Spatio-spectral
# false-colour plotting" page of the NIFTy manual.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import nifty.re as jft

# The projector is framework-agnostic — it consumes and returns plain numpy
# arrays — so a nifty.re demo imports it straight from nifty.cl.
from nifty.cl.spectrum_to_rgb import SpectrumToRGBProjector

jax.config.update("jax_enable_x64", True)

# %%
nx, ny = 256, 256           # spatial pixels
n_spec = 16                 # spectral bins, after cropping
n_spec_padded = n_spec + n_spec//2

key = jax.random.PRNGKey(43)

# %% [markdown]
# ## A field whose spectrum varies from pixel to pixel
#
# The single 3-D `add_fluctuations` call is essential. Two separate calls, one
# spatial and one spectral, would give a separable log-field
#
#     log_signal(x, y, k) = spatial(x, y) + spectral(k)
#
# in which every pixel shares one spectral shape up to an amplitude — and the
# false-colour image collapses to greyscale.
#
# The spectral axis is drawn 50 % longer than needed and cropped afterwards.
# The correlated field is periodic, so without that padding the first and last
# bins wrap into one another and the red and blue ends come out near-identical.
#
# The fluctuation amplitude is deliberately large: exponentiating it gives the
# field a luminance range of nearly two decades, which is what makes the choice
# of tone curve below worth making.

# %%
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=0., offset_std=(1e-3, 1e-32))
cfm.add_fluctuations(
    (nx, ny, n_spec_padded),
    distances=(1./nx, 1./ny, 1./n_spec_padded),
    fluctuations=(2., 1e-32),
    loglogavgslope=(-3., 1e-32),
    prefix="cf",
    non_parametric_kind="power",
)
cf = cfm.finalize()

key, subkey = jax.random.split(key)
signal = np.array(jnp.exp(cf(jft.random_like(subkey, cf.domain))))[:, :, :n_spec]

# %% [markdown]
# ## Projection
#
# `spectral_axis_type='energy'` sends bin 0 to the red end and the last bin to
# the blue end; `visible_bin_width='uniform'` gives every bin the same visible
# wavelength width, which suits equi-spaced index bins. The luminance range is
# asked of the data, with the black point at the 30th percentile so that the
# faint background stays dark.

# %%
bin_width = 1.
proj = SpectrumToRGBProjector(flux_convention='bin_integrated_flux',
                              spectral_axis_type='energy',
                              visible_bin_width='uniform')
proj.specify_input_spectrum_bins_via_center_and_width(
    bin_width*(0.5 + np.arange(n_spec)), np.full(n_spec, bin_width))

black, white = proj.luminance_quantiles(signal, q=(0.3, 0.999))
proj.set_luminance_range(white=white, black=black)

# %% [markdown]
# ## Rendering, with its legend
#
# In the colour map, row *i* and column *k* show how a spectrum carrying flux
# `levels[i]` in bin *k* alone is rendered. A single-bin spectrum is far dimmer
# than a broadband one at the same flux per bin, so the levels are obtained by
# inverting the luminance range through the per-bin response rather than
# guessed — guessing them is the usual way to end up with an all-black legend.
#
# The colour map follows the tone curve of the most recent projection, so each
# legend below genuinely belongs to the panel above it.

# %%
per_bin_luminance = proj.luminance_of_spectrum(np.eye(n_spec))
levels = np.geomspace(black/per_bin_luminance.max(),
                      white/per_bin_luminance.min(), 128)

renderings = []
for log_compression in (False, True):
    proj.use_log_compression(log_compression)
    renderings.append((proj.project(signal),
                       proj.get_color_map_image(levels=levels),
                       "logarithmic" if log_compression else "linear"))

# %%
fig = plt.figure(figsize=(10, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], hspace=0.45, wspace=0.08)

for col, (rgb, legend, label) in enumerate(renderings):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(rgb.transpose(1, 0, 2), origin='lower', extent=[0, 1, 0, 1])
    ax.set_title(f"{label} luminance mapping", fontsize=11)
    ax.set_xlabel("x")
    if col == 0:
        ax.set_ylabel("y")
    else:
        ax.set_yticklabels([])

    ax_cb = fig.add_subplot(gs[1, col])
    ax_cb.imshow(legend, origin='lower', aspect='auto',
                 extent=[0.5, n_spec + 0.5, 0., 1.])
    ax_cb.set_title(f"colour map: bin 1 = red end, bin {n_spec} = blue end",
                    fontsize=8, pad=3)
    ax_cb.set_xlabel("spectral bin index", fontsize=8)
    ax_cb.set_xticks(np.arange(2, n_spec + 1, 2))
    ax_cb.tick_params(labelsize=7)
    if col == 0:
        # both legends span the same flux levels, so one labelled axis is enough
        ax_cb.set_ylabel("flux", fontsize=8)
        proj.apply_color_map_ticks(ax_cb, axis='y', n_ticks=4)
    else:
        ax_cb.set_yticks([])

fig.savefig("a_spatio-spectral_plotting.png", dpi=150, bbox_inches='tight')
plt.show()
