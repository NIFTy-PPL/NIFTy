#!/usr/bin/env python3

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2025 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

"""Demo: spatio-spectral false-colour visualization (nifty.cl).

Equivalent to demos/re/a_spatio-spectral_plotting.py but using nifty.cl and
plain numpy for field generation.

nifty.cl's CorrelatedFieldMaker is an inference operator, not a sampler, so
the correlated field is drawn directly via spectral synthesis (FFT):
  1. Draw Gaussian white noise in real space.
  2. Fourier-transform to k-space and apply a power-law amplitude filter
     A(k) ∝ |k|^(slope/2) with slope = -3 (power kind), matching the
     loglogavgslope used in the nifty.re demo.
  3. Inverse-FFT back to real space and normalise the RMS.
  4. Exponentiate to obtain a strictly positive log-normal field.

Using a single 3-D FFT is essential for the same reason as in the nifty.re
demo: a separable construction would make the spectral shape identical at
every pixel, collapsing the false-colour image to greyscale.

The spectral axis is padded by +50 % before synthesis and cropped afterwards
to avoid periodic-boundary-condition wrap-around artefacts (which would
otherwise make the red and blue slice images near-identical).

For a single-panel false-colour plot nifty.cl's Plot class can be used
directly::

    import nifty.cl as ift
    p = ift.Plot()
    p.add(field, spectral_axis_type='energy', visible_bin_width='uniform',
          dynamic_range=10)
    p.output(name='out.png')

This demo instead builds a multi-panel figure manually to also show the
spectral slices and the colour-map legend.
"""

import matplotlib.pyplot as plt
import numpy as np
import nifty.cl as ift
from matplotlib.colors import LogNorm

from nifty.cl.spectrum_to_rgb import SpectrumToRGBProjector

# ---------------------------------------------------------------------------
# Field dimensions
# ---------------------------------------------------------------------------
nx, ny = 256, 256           # spatial pixels
n_spec = 16                 # spectral bins (after de-padding)
n_spec_padded = n_spec + n_spec // 2   # +50 % to avoid periodic BC wrap-around

seed = 42
rng = np.random.default_rng(seed)

use_log_compression = True

# ---------------------------------------------------------------------------
# Correlated log-normal field via spectral synthesis
#
# P(k) ∝ |k|^slope  (power kind, matching loglogavgslope = -3 in nifty.re).
# The same slope applies isotropically to all three dimensions so that the
# spectral shape genuinely varies from pixel to pixel.
# ---------------------------------------------------------------------------
white = rng.standard_normal((nx, ny, n_spec_padded))
white_k = np.fft.fftn(white, norm='ortho')

kx = np.fft.fftfreq(nx) * nx          # mode indices 0, 1, …, nx//2
ky = np.fft.fftfreq(ny) * ny          # matches nifty.re distances=1/N convention:
ks = np.fft.fftfreq(n_spec_padded) * n_spec_padded  # k_n = n/(N·d) = n for d=1/N
KX, KY, KS = np.meshgrid(kx, ky, ks, indexing='ij')
K = np.sqrt(KX**2 + KY**2 + KS**2)
K[0, 0, 0] = 1.0   # avoid division by zero at DC

slope = -3.0                       # log-log power-spectrum slope
A = K ** (slope / 2.0)             # amplitude filter: A ∝ sqrt(P) ∝ k^(slope/2)
A[0, 0, 0] = 0.0                   # zero DC → zero-mean log-field

log_signal = np.real(np.fft.ifftn(white_k * A, norm='ortho'))
log_signal *= 0.85 / log_signal.std()   # normalise RMS (matches fluctuations=0.85)

signal = np.exp(log_signal[:, :, :n_spec])   # de-pad; shape: (nx, ny, n_spec)

# ---------------------------------------------------------------------------
# Wrap in a nifty.cl Field
#
# Domain convention (matches nifty.cl plot.py, freq_space_idx=1 default):
#   domain[0] = 2-D spatial RGSpace
#   domain[1] = 1-D spectral RGSpace
# ---------------------------------------------------------------------------
spatial_dom = ift.RGSpace((nx, ny), distances=1. / nx)
spectral_dom = ift.RGSpace(n_spec, distances=1.)
domain = ift.DomainTuple.make([spatial_dom, spectral_dom])
field = ift.makeField(domain, signal)   # noqa: F841  (used by ift.Plot alternative above)

# ---------------------------------------------------------------------------
# SpectrumToRGBProjector
#
# spectral_axis_type='energy': bin 0 (lowest energy) → red end (~634 nm),
#                              bin n_spec-1 (highest) → blue end (~446 nm).
# visible_bin_width='uniform': every bin receives the same visible Δλ.
# ---------------------------------------------------------------------------
bin_width = 1.0
centers = bin_width * (0.5 + np.arange(n_spec))
widths = np.full(n_spec, bin_width)

proj = SpectrumToRGBProjector(
    spectral_axis_type='energy',
    visible_bin_width='uniform',
)
proj.specify_input_spectrum_bins_via_center_and_width(centers, widths)

# ---------------------------------------------------------------------------
# Project spatio-spectral field to sRGB
# ---------------------------------------------------------------------------
rgb_flat = proj.project_total_spectral_bin_flux(
    signal.reshape(-1, n_spec),
    dynamic_range=10 if use_log_compression else None,
)
rgb_image = rgb_flat.reshape(nx, ny, 3)   # (nx, ny, 3)

# ---------------------------------------------------------------------------
# Colour map legend
#
# get_color_map_image returns (n_levels, n_spec, 3): each row is a flux level
# (dim → bright) and each column is a spectral bin (red → blue).
# ---------------------------------------------------------------------------
n_levels = 64
cmap_max = 1.0
if use_log_compression:
    cmap_min = 1e-2
    levels = np.geomspace(cmap_min, cmap_max, n_levels)
else:
    cmap_min = 0.0
    levels = np.linspace(cmap_min, cmap_max, n_levels)
colormap_image = proj.get_color_map_image(total_flux_levels=levels)
# shape: (n_levels, n_spec, 3)

# ---------------------------------------------------------------------------
# Spectral slices: bins at red, green, and blue ends of the visible range.
# With energy mode and the default 440–640 nm mapping each bin spans
# 200/n_spec nm; centres at approximately:
#   bin 0       → ~634 nm  (red)
#   bin n//2    → ~534 nm  (green, near the CIE luminous-efficiency peak)
#   bin n_spec-1 → ~446 nm (blue)
# ---------------------------------------------------------------------------
slice_bins = [0, n_spec // 2, n_spec - 1]
slice_labels = ["red end (~634 nm)", "green (~534 nm)", "blue end (~446 nm)"]
log_vmin = signal.min()
log_vmax = signal.max()

# ---------------------------------------------------------------------------
# Figure layout
#   row 0 (large)  : false-colour image
#   row 1 (medium) : 3 spectral slices at primary colours
#   row 2 (thin)   : colour map legend
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(
    3, 3,
    height_ratios=[5, 2.5, 0.6],
    hspace=0.35,
    wspace=0.08,
)

# -- Main false-colour image -------------------------------------------------
ax_img = fig.add_subplot(gs[0, :])
ax_img.imshow(
    rgb_image.transpose(1, 0, 2),   # (ny, nx, 3) with y upward
    origin='lower',
    extent=[0, 1, 0, 1],
)
ax_img.set_title(r"Spatio-spectral log-normal field", fontsize=12)
ax_img.set_xlabel("x")
ax_img.set_ylabel("y")

# -- Spectral slices ---------------------------------------------------------
norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
ax_slices = []
for col, (k, label) in enumerate(zip(slice_bins, slice_labels)):
    ax = fig.add_subplot(gs[1, col])
    im = ax.imshow(
        signal[:, :, k].T,
        origin='lower',
        extent=[0, 1, 0, 1],
        cmap='inferno',
        norm=norm,
    )
    ax.set_title(f"Bin {k + 1} — {label}", fontsize=9)
    ax.set_xlabel("x", fontsize=8)
    if col == 0:
        ax.set_ylabel("y", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7)
    ax_slices.append(ax)

fig.colorbar(im, ax=ax_slices, orientation='vertical',
             fraction=0.02, pad=0.01, label="signal (log scale)")

# -- Colour map legend -------------------------------------------------------
ax_cb = fig.add_subplot(gs[2, :])
ax_cb.imshow(
    colormap_image,
    origin='lower',
    aspect='auto',
    extent=[0.5, n_spec + 0.5, 0, 1],
)
ax_cb.set_title(
    "Colour map legend: spectral bin  ×  relative flux  "
    r"(bin 1 = red end, bin {:d} = blue end)".format(n_spec),
    fontsize=9, pad=3,
)
ax_cb.set_xlabel("spectral bin index", fontsize=9)
ax_cb.set_ylabel("rel.\nflux", fontsize=7, rotation=0, labelpad=28)
ax_cb.set_xticks(np.arange(1, n_spec + 1))
ax_cb.tick_params(axis='x', labelsize=7)
ax_cb.yaxis.set_ticks([0, 0.5, 1])
if use_log_compression:
    cmap_middle = np.exp((np.log(cmap_min) + np.log(cmap_max)) / 2)
    fmt = lambda s: f'{s:1.1e}'
else:
    cmap_middle = (cmap_min + cmap_max) / 2
    fmt = lambda s: f'{s:1.1f}'
ax_cb.yaxis.set_ticklabels(
    [fmt(s) for s in [cmap_min, cmap_middle, cmap_max]],
    fontsize=7,
)

fig.savefig("a_spatio-spectral_plotting.png", dpi=150, bbox_inches='tight')
plt.show()
