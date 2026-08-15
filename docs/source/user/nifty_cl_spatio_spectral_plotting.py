# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Spatio-spectral false-colour plotting

# A field with a spectral axis is a stack of images, one per spectral bin. Looking at the bins
# one at a time hides how the spectrum varies from pixel to pixel; averaging them away discards
# that information entirely. `SpectrumToRGBProjector` instead maps the spectral axis onto the
# visible light range and converts it through the CIE 1931 model of human colour perception, so
# that a single image encodes **spectral shape as hue** and **flux as brightness**.
#
# The projector consumes and returns plain `numpy` arrays: you hand it an array whose last axis
# is the spectral one and get back an sRGB array of shape `(..., 3)`, which you can plot however
# you like. It is therefore equally usable from `nifty.cl`, from `nifty.re`, or with no NIFTy
# field involved at all.
#
# Signatures and per-argument details live in the
# {doc}`API reference <../mod/nifty.cl.spectrum_to_rgb>`; this page is about which of them you
# need and why.

# ## Which section do you need?
#
# * *I just want an image on screen* — **A minimal working example**, below.
# * *My data are flux densities, not fluxes per bin* — **Setting up the projector**.
# * *Two panels of the same source look equally bright although one is fainter* —
#   **Deciding what black and white mean**.
# * *My image is almost all black, or washed out* — **Deciding what black and white mean** and
#   **Linear or logarithmic**.
# * *Bright regions look wrong / lose their colour* — **What happens above the white point**.
# * *I need a legend telling readers which colour means what* — **The colour-map legend**.
# * *Why these colours?* — **Appendix: how it works**.

# +
import logging
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import nifty.cl as ift
from nifty.cl.spectrum_to_rgb import ColorSpaceTools, SpectrumToRGBProjector

# -

# ## A toy data cube
#
# Three sources with clearly different spectra, on a faint broadband background. One source is a
# hundred times brighter than the other two, which is what makes the tone-mapping sections below
# worth reading. The spectral axis is an *energy* axis with 24 equally wide bins.
#
# For realistic examples built from correlated fields, see
# `demos/cl/a_spatio-spectral_plotting.py` and `demos/re/a_spatio-spectral_plotting.py`.

# +
nx, ny, n_spec = 96, 96, 24

bin_width = 1.
bin_centers = bin_width*(0.5 + np.arange(n_spec))
bin_widths = np.full(n_spec, bin_width)


def blob(cx, cy, width=9.):
    x = np.arange(nx)[:, np.newaxis]
    y = np.arange(ny)[np.newaxis, :]
    return np.exp(-((x - cx)**2 + (y - cy)**2)/(2*width**2))


def spectral_peak(center, width=3.):
    return np.exp(-(np.arange(n_spec) - center)**2/(2*width**2))


cube = (100.*blob(28, 30)[..., np.newaxis]*spectral_peak(3.)     # low energy, very bright
        + 1.*blob(66, 34)[..., np.newaxis]*spectral_peak(12.)    # mid energy
        + 1.*blob(46, 68)[..., np.newaxis]*spectral_peak(20.)    # high energy
        + 0.02)                                                  # faint broadband background

cube.shape
# -


# A helper to keep the plotting below out of the way. Note the transpose: the projector leaves
# the spatial axes untouched, and `imshow` wants them the other way round.

def show(rgb, title, ax=None):
    ax = plt.subplots(figsize=(3.6, 3.6))[1] if ax is None else ax
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# ## A minimal working example
#
# Four steps: say what your data are, say where the bins are, say which luminances map to black
# and white, and project.

# +
proj = SpectrumToRGBProjector(spectral_axis_type='energy',
                              visible_bin_width='uniform',
                              flux_convention='bin_integrated_flux')
proj.specify_input_spectrum_bins_via_center_and_width(bin_centers, bin_widths)

black, white = proj.luminance_quantiles(cube, q=(0.5, 0.999))
proj.set_luminance_range(white=white, black=black)

rgb = proj.project(cube)
show(rgb, "low energy = red, high energy = blue")
# -

# The low-energy source came out red and the high-energy one blue because `spectral_axis_type`
# is `'energy'`; the projector inverts the axis, consistent with $E = hc/\lambda$. The middle
# source is green because it sits near the peak of the eye's luminous efficiency — which is also
# why it looks brighter than the blue source despite carrying the same flux.
#
# Every projection logs the range it used. That is deliberate — it is the quickest way to see
# whether two figures were rendered comparably. To keep the rest of this page short we raise the
# log level now, which leaves warnings visible.

ift.logger.setLevel(logging.WARNING)

# ## Setting up the projector
#
# The three constructor arguments are all mandatory-in-spirit decisions. Each one prevents a
# specific failure, and none of them can be detected from the data.
#
# **`flux_convention`** — `'bin_integrated_flux'` if your values are fluxes $\Phi_k$ already
# integrated over their bin, `'flux_density'` if they are densities
# $\partial\Phi/\partial[E\,|\,\lambda]$ that still need multiplying by the bin widths. Pick the
# wrong one with non-uniform bins and you get a plausible image with the wrong colours, not an
# error. The projector announces its choice when constructed for exactly this reason.
#
# **`spectral_axis_type`** — `'energy'` maps low input to the red end (direction-inverting);
# `'wavelength'` maps low input to the blue end. Only the direction is affected.
#
# **`visible_bin_width`** — `'uniform'` gives every bin the same visible $\Delta\lambda$
# regardless of its width in your domain, which is what you usually want for energy bins,
# especially geometrically spaced ones. `'proportional'` makes each bin's visible width follow
# its input width, preserving relative spectral coverage; natural for wavelength data.
#
# Bins can be given as centres and widths, as above, or as explicit boundaries — convenient for
# geometrically spaced bins, and the only option when bins are not contiguous:

geom_boundaries = np.geomspace(1., 100., n_spec + 1)
geom_proj = SpectrumToRGBProjector('energy', 'uniform', 'flux_density')
geom_proj.specify_input_spectrum_bins_via_bin_boundaries(geom_boundaries[:-1],
                                                         geom_boundaries[1:])
geom_proj.flux_convention

# ## Deciding what black and white mean
#
# This is the part worth spending time on. The projector maps a luminance range onto the
# displayable range: everything at or below `black` renders black, everything at or above
# `white` renders at full brightness. Those two numbers are the entire brightness contract.
#
# If you never set them, each image is normalised to its own maximum — so a source that halves
# in brightness looks *identical*, because the scale silently halved with it:

# +
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
for ax, (factor, label) in zip(axes, [(1., "cube"), (0.5, "cube/2")]):
    auto_proj = SpectrumToRGBProjector('energy', 'uniform', 'bin_integrated_flux')
    auto_proj.specify_input_spectrum_bins_via_center_and_width(bin_centers, bin_widths)
    show(auto_proj.project(factor*cube), f"auto white point: {label}", ax=ax)
fig.tight_layout()
# -

# Hence the warning the projector emits when no range is set. Fix it by fixing the range, which
# you can do in four ways — all of which resolve to the same pair of numbers, so they mix freely.
#
# **From quantiles of the data** — the usual choice, and the one used above. It asks the data
# where its interesting luminances lie instead of making you guess absolute numbers. Zero and
# negative luminances are excluded, so masked or padded fields do not drag the lower quantile to
# zero. For log rendering, a lower quantile near the background level (~0.3–0.5) usually works
# better than the default 0.01.

proj.luminance_quantiles(cube, q=(0.5, 0.999))

# **From a reference spectrum** — when you can say physically what should be white. Build the
# spectrum you mean and ask what luminance it produces:

# +
flat_spectrum = 5.*bin_widths/bin_widths.sum()   # a flat spectrum carrying a total flux of 5
white_from_spectrum = proj.luminance_of_spectrum(flat_spectrum)

proj.set_luminance_range(white=white_from_spectrum, black=0.)
show(proj.project(cube), "white point = flat spectrum, total flux 5")
# -

# **From a dynamic range** — places the black point below the white point,
# `black = white/dynamic_range`, whichever way the white point was obtained.
#
# **Explicitly** — pass the two luminances directly, e.g. to reuse a range computed elsewhere.

proj.set_luminance_range(white=white, dynamic_range=100.)
proj.luminance_range

# The resolved pair is available as `luminance_range`, and applies to every subsequent
# projection until you change it. That is what makes a series of panels comparable — and what
# lets the colour-map legend agree with the image.

# ## Linear or logarithmic
#
# By default the curve between black and white is linear. For data spanning decades — our
# hundredfold source ratio, for example — logarithmic compression shows faint and bright
# structure at once. A black point is then mandatory: the curve is
# $\log(Y/Y_\mathrm{black})/\log(Y_\mathrm{white}/Y_\mathrm{black})$, which is undefined at zero,
# and projecting with log compression and no black point raises.

# +
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))

proj.set_luminance_range(white=white, black=black)
proj.use_log_compression(False)
show(proj.project(cube), "linear", ax=axes[0])

proj.use_log_compression()
show(proj.project(cube), "logarithmic", ax=axes[1])
fig.tight_layout()
# -

# Note that a black point is *not* tied to log compression: it applies to the linear curve too,
# where it acts exactly like `vmin` in matplotlib.

# ## What happens above the white point
#
# Pixels brighter than the white point have to give something up, and you choose what.
#
# `highlights='clamp'` (the default) clamps the luminance, which preserves chromaticity: an
# over-bright region keeps its hue and loses internal detail. `highlights='clip_channels'` lets
# the individual sRGB channels clip independently, so a blowing-out region drifts in hue the way
# an over-exposed camera sensor does.
#
# The difference only appears *above* the white point, so we deliberately place the white point
# well below the bright source here — and switch back to a linear curve, since log compression
# pulls almost everything back inside the range.

# +
proj.use_log_compression(False)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
for ax, mode in zip(axes, ['clamp', 'clip_channels']):
    proj.set_luminance_range(white=white/20., black=black, highlights=mode)
    show(proj.project(cube), f"highlights='{mode}'", ax=ax)
fig.tight_layout()
# -

# Under `'clamp'` the over-exposed core keeps the orange of its spectrum; under
# `'clip_channels'` the red channel saturates first and the core turns yellow. Neither is more
# correct than the other — `'clamp'` is honest about hue, `'clip_channels'` looks photographic.

# ## The colour-map legend
#
# `get_color_map_image` renders the projector's own key: row *i*, column *k* shows how a
# spectrum carrying flux `levels[i]` in bin *k* **alone** is rendered. It uses the tone curve of
# the most recent projection, so the legend cannot disagree with the image it accompanies.
#
# Nothing is rescaled per column. Since luminance per unit flux varies by roughly a factor of
# ten across the visible range, mid-spectrum columns are genuinely brighter than the red and
# blue ends at equal flux, and the columns reach white at different heights. The legend shows
# that rather than hiding it.
#
# You choose the flux levels, and they generally need to reach **far beyond the per-bin fluxes
# in your data**: a single-bin spectrum is much dimmer than a broadband one carrying the same
# flux per bin. Choosing them by eye is how you end up with an all-black legend. Instead, invert
# the luminance range through the per-bin response, which `luminance_of_spectrum` gives you
# directly for one-hot spectra:

# +
per_bin_luminance = proj.luminance_of_spectrum(np.eye(n_spec))
level_min = black/per_bin_luminance.max()    # brightest bin just leaves the black point
level_max = white/per_bin_luminance.min()    # faintest bin finally reaches the white point

print(f"single-bin spectra span the tone curve from {level_min:.1f} to {level_max:.0f} flux "
      f"units, against a maximum of {cube.max():.0f} per bin in the data itself")
# -

# +
proj.set_luminance_range(white=white, black=black)
proj.use_log_compression()
proj.project(cube)                                   # sets the curve the legend will use

levels = np.geomspace(level_min, level_max, 128)
legend = proj.get_color_map_image(levels=levels)

fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.imshow(legend, origin='lower', aspect='auto', extent=[0.5, n_spec + 0.5, 0., 1.])
proj.apply_color_map_ticks(ax, axis='y', n_ticks=4)
ax.set_xlabel("spectral bin index")
ax.set_ylabel("flux")
ax.set_title("bin 1 = red end, bin 24 = blue end", fontsize=9)
fig.tight_layout()
# -

# `apply_color_map_ticks` reads the extent back off the axis, so the `extent` and `origin` you
# drew with are respected. If you would rather place the ticks yourself, `get_color_map_ticks`
# returns the positions and labels instead of applying them.

# ## If you already use `ift.Plot`
#
# `Plot` renders a two-space field with a spectral `RGSpace` in false colour, configured through
# a single `color_mapping_kwargs` dictionary whose keys are named after the projector arguments
# used above:

# +
domain = ift.DomainTuple.make([ift.RGSpace((nx, ny), distances=1./nx),
                               ift.RGSpace(n_spec, distances=bin_width)])
field = ift.makeField(domain, cube)

plot = ift.Plot()
plot.add(field, color_mapping_kwargs=dict(spectral_axis_type='energy',
                                          visible_bin_width='uniform',
                                          flux_convention='bin_integrated_flux',
                                          quantiles=(0.5, 0.999),
                                          log_compression=True))
plot.output(name=os.path.join(tempfile.mkdtemp(), "spatio_spectral_plot.png"),
            xsize=6, ysize=6)
# -

# This route exists mainly for backwards compatibility, and it decides several things on your
# behalf: bins are derived from the spectral space's `distances` as uniform bins starting at
# zero, the domain must be a two-entry `DomainTuple`, and omitted setup arguments fall back to
# defaults with a warning. If any of that does not describe your data, drive the projector
# directly as in the sections above — it is only a few lines more.

# ## `ColorSpaceTools`
#
# The projector's colour-science machinery is a separate class of static helpers:
# conversions between XYZ, xyY and LMS, the CIE 1931 standard observer and D65 illuminant
# tables, and the sRGB embedding. Most users never touch it. The one knob worth knowing is
# `enhance_sRGB_color_contrast`, which increases colour saturation of a finished sRGB image
# while leaving its black and white points alone:

vivid = ColorSpaceTools.enhance_sRGB_color_contrast(proj.project(cube), 1.8)
show(vivid, "colour contrast x1.8")

# ## Appendix: how it works
#
# Four steps happen inside `project`.
#
# **1. The spectral axis is mapped onto the visible range.** Each input bin is assigned a
# wavelength interval between `wavelength_min_mappable` and `wavelength_max_mappable`
# (440–640 nm by default), with the direction set by `spectral_axis_type` and the widths by
# `visible_bin_width`. The default range is narrower than the full visible span because
# saturated spectral colours near its edges lie far outside the sRGB gamut.
#
# **2. Bin fluxes become tristimulus values.** For each visible bin the CIE 1931 colour-matching
# functions are averaged over the bin, giving a fixed `(n_bins, 3)` tensor; contracting your data
# with it evaluates $XYZ = \int P(\lambda)\,\mathrm{CMF}(\lambda)\,\mathrm{d}\lambda$ for an
# emissive source. The $Y$ component is luminance — the quantity every part of the brightness
# interface above is expressed in, and the reason `luminance_of_spectrum` exists.
#
# **3. Luminance is tone-mapped.** The linear or logarithmic curve maps
# $[Y_\mathrm{black}, Y_\mathrm{white}]$ onto $[0, 1]$, and all three of $X$, $Y$, $Z$ are scaled
# by the same factor $L/Y$, so only brightness changes and hue is preserved. The lower end always
# clamps at zero; the upper end obeys `highlights`.
#
# **4. XYZ is embedded in sRGB.** The standard D65 matrix and gamma correction, followed by a
# per-channel clip to $[0, 1]$.
#
# That final clip is a known limitation. Strongly monochromatic spectra fall outside the sRGB
# gamut and produce negative channel values, which clipping turns into a slight hue shift and a
# brightness that no longer matches the luminance the tone curve asked for. Replacing it with a
# proper gamut mapping — desaturating toward the white point at constant luminance — is planned.
