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

import logging

import numpy as np
import pytest

from nifty.cl.logger import logger
from nifty.cl.spectrum_to_rgb import ColorSpaceTools, SpectrumToRGBProjector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_projector(n_bins=100, bin_width=1.0, flux_convention='bin_integrated_flux',
                    spectral_axis_type='energy', visible_bin_width='uniform'):
    centers = bin_width * (0.5 + np.arange(n_bins))
    widths = np.full(n_bins, bin_width)
    proj = SpectrumToRGBProjector(flux_convention=flux_convention,
                                  spectral_axis_type=spectral_axis_type,
                                  visible_bin_width=visible_bin_width)
    proj.specify_input_spectrum_bins_via_center_and_width(centers, widths)
    return proj, bin_width


class _RecordingHandler(logging.Handler):
    """Capture NIFTy log records; caplog cannot, since the logger does not propagate."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _capture_nifty_logs():
    handler = _RecordingHandler()
    logger.addHandler(handler)
    return handler


# ---------------------------------------------------------------------------
# ColorSpaceTools
# ---------------------------------------------------------------------------

def test_xyz_to_xyY_roundtrip():
    xyz = np.array([[0.3, 0.5, 0.2], [1.0, 1.0, 1.0]])
    assert np.allclose(ColorSpaceTools.xyY_to_XYZ(ColorSpaceTools.XYZ_to_xyY(xyz)), xyz)


def test_xyz_to_xyY_black_pixel():
    black = np.zeros((1, 3))
    xyY = ColorSpaceTools.XYZ_to_xyY(black)
    xyz_back = ColorSpaceTools.xyY_to_XYZ(xyY)
    assert np.allclose(xyz_back, 0.)


def test_xyz_to_lms_roundtrip():
    xyz = np.array([[0.3, 0.5, 0.2], [1.0, 1.0, 1.0]])
    assert np.allclose(ColorSpaceTools.LMS_to_XYZ(ColorSpaceTools.XYZ_to_LMS(xyz)), xyz,
                       atol=1e-10)


def test_embed_xyz_black_to_black():
    black = np.zeros((1, 3))
    assert np.allclose(ColorSpaceTools.embed_XYZ_perceived_color_in_sRGB(black), 0.)


def test_embed_xyz_d65_white_to_white():
    # D65 white point in XYZ (CIE standard)
    d65_white = np.array([[0.9505, 1.0000, 1.0890]])
    srgb = ColorSpaceTools.embed_XYZ_perceived_color_in_sRGB(d65_white)
    assert np.allclose(srgb, 1., atol=1e-2)


def test_cie1931_peak_luminous_efficiency():
    # Y of the CIE 1931 observer peaks at ~555 nm (value ≈ 1)
    xyz = ColorSpaceTools.get_cie1931_standard_observer_XYZ_tristimulus_values(
        np.array([555.]))
    assert xyz[0, 1] > 0.99


def test_cie1931_out_of_range_clamped():
    # Values outside 380–780 nm should be clamped to the table endpoints (near zero)
    xyz_low = ColorSpaceTools.get_cie1931_standard_observer_XYZ_tristimulus_values(
        np.array([200.]))
    xyz_high = ColorSpaceTools.get_cie1931_standard_observer_XYZ_tristimulus_values(
        np.array([900.]))
    assert np.all(xyz_low < 0.01)
    assert np.all(xyz_high < 0.01)


# ---------------------------------------------------------------------------
# Setup / validation
# ---------------------------------------------------------------------------

def test_flux_convention_is_required_and_validated():
    with pytest.raises(TypeError):
        SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform')
    with pytest.raises(ValueError):
        SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform',
                               flux_convention='total_flux')


def test_flux_convention_is_logged_at_construction():
    handler = _capture_nifty_logs()
    try:
        SpectrumToRGBProjector('flux_density', 'energy', 'uniform')
    finally:
        logger.removeHandler(handler)
    assert any("flux density" in r.getMessage() for r in handler.records)


def test_spectral_axis_invalid_raises():
    with pytest.raises(ValueError):
        SpectrumToRGBProjector('bin_integrated_flux', 'invalid', 'uniform')


def test_visible_bin_width_invalid_raises():
    with pytest.raises(ValueError):
        SpectrumToRGBProjector('bin_integrated_flux', 'energy', 'invalid')


def test_bin_boundary_spec_rejects_overlap():
    proj = SpectrumToRGBProjector('bin_integrated_flux', 'energy', 'uniform')
    lower = np.array([0., 1., 1.5])
    upper = np.array([1.2, 2., 2.5])  # bins 0 and 1 overlap
    with pytest.raises(ValueError):
        proj.specify_input_spectrum_bins_via_bin_boundaries(lower, upper)


def test_bin_spec_rejects_mismatched_lengths():
    proj = SpectrumToRGBProjector('bin_integrated_flux', 'energy', 'uniform')
    with pytest.raises(ValueError):
        proj.specify_input_spectrum_bins_via_bin_boundaries(
            np.array([0., 1.]), np.array([1., 2., 3.]))


def test_projection_rejects_negative_flux():
    proj, _ = _make_projector()
    with pytest.raises(ValueError):
        proj.project(-np.ones((1, 100)))


def test_projection_rejects_wrong_n_bins():
    proj, _ = _make_projector(n_bins=100)
    with pytest.raises(ValueError):
        proj.project(np.ones((1, 50)))


def test_respecifying_bins_invalidates_luminance_range():
    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=1.)
    assert proj.luminance_range is not None
    proj.specify_input_spectrum_bins_via_bin_boundaries(np.arange(10.), np.arange(10.) + 1.)
    assert proj.luminance_range is None


# ---------------------------------------------------------------------------
# Luminance converters
# ---------------------------------------------------------------------------

def test_luminance_of_spectrum_is_linear_in_flux():
    proj, _ = _make_projector(n_bins=10)
    spectrum = np.ones(10)
    Y = proj.luminance_of_spectrum(spectrum)
    assert Y > 0.
    assert np.isclose(proj.luminance_of_spectrum(3.*spectrum), 3.*Y)


def test_luminance_of_spectrum_matches_projection_luminance():
    proj, _ = _make_projector(n_bins=10)
    rng = np.random.default_rng(3)
    data = rng.uniform(0., 1., (4, 10))
    captured = {}
    proj.set_luminance_range(Y_saturation=1.)
    proj.project(data, XYZ_inspect_callback=lambda raw, mapped: captured.update(raw=raw))
    assert np.allclose(captured['raw'][..., 1], proj.luminance_of_spectrum(data))


def test_luminance_of_spectrum_honours_flux_convention():
    bw = 0.5
    proj_int, _ = _make_projector(n_bins=10, bin_width=bw)
    proj_den, _ = _make_projector(n_bins=10, bin_width=bw,
                                  flux_convention='flux_density')
    spectrum = np.ones(10)
    assert np.isclose(proj_den.luminance_of_spectrum(spectrum),
                      proj_int.luminance_of_spectrum(spectrum*bw))


def test_luminance_quantiles_excludes_non_positive():
    proj, _ = _make_projector(n_bins=10)
    bright = np.ones((10, 10))
    data = np.concatenate([np.zeros((90, 10)), bright], axis=0)
    lo, hi = proj.luminance_quantiles(data, q=(0.0, 1.0))
    Y_bright = proj.luminance_of_spectrum(np.ones(10))
    # without the exclusion the lower quantile would be 0 (90% of pixels are zero)
    assert np.isclose(lo, Y_bright) and np.isclose(hi, Y_bright)


def test_luminance_quantiles_returns_requested_quantiles():
    proj, _ = _make_projector(n_bins=10)
    levels = np.linspace(1., 100., 1000)
    data = levels[:, np.newaxis]*np.ones(10)[np.newaxis, :]
    lo, hi = proj.luminance_quantiles(data, q=(0.1, 0.9))
    Y = proj.luminance_of_spectrum(np.ones(10))
    assert np.isclose(lo, np.quantile(levels, 0.1)*Y, rtol=1e-6)
    assert np.isclose(hi, np.quantile(levels, 0.9)*Y, rtol=1e-6)


def test_luminance_quantiles_rejects_bad_q():
    proj, _ = _make_projector(n_bins=10)
    with pytest.raises(ValueError):
        proj.luminance_quantiles(np.ones((4, 10)), q=(0.9, 0.1))


def test_luminance_quantiles_all_zero_raises():
    proj, _ = _make_projector(n_bins=10)
    with pytest.raises(ValueError):
        proj.luminance_quantiles(np.zeros((4, 10)))


# ---------------------------------------------------------------------------
# set_luminance_range
# ---------------------------------------------------------------------------

def test_set_luminance_range_is_keyword_only():
    proj, _ = _make_projector(n_bins=10)
    with pytest.raises(TypeError):
        proj.set_luminance_range(1., 0.1)


def test_Y_black_defaults_to_zero():
    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=2.)
    assert proj.luminance_range == (0., 2.)


def test_dynamic_range_places_black_point():
    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=10., dynamic_range=100.)
    assert proj.luminance_range == (0.1, 10.)


def test_black_and_dynamic_range_are_mutually_exclusive():
    proj, _ = _make_projector(n_bins=10)
    with pytest.raises(ValueError):
        proj.set_luminance_range(Y_saturation=10., Y_black=1., dynamic_range=10.)


@pytest.mark.parametrize('kwargs', [
    dict(Y_saturation=1., Y_black=2.),        # black above white
    dict(Y_saturation=1., Y_black=1.),        # degenerate
    dict(Y_saturation=1., Y_black=-1.),       # negative
    dict(Y_saturation=-1.),                 # non-positive white
    dict(Y_saturation=1., dynamic_range=0.5),   # dynamic range below 1
    dict(Y_saturation=1., highlights='nonsense'),
])
def test_set_luminance_range_validation(kwargs):
    proj, _ = _make_projector(n_bins=10)
    with pytest.raises(ValueError):
        proj.set_luminance_range(**kwargs)


# ---------------------------------------------------------------------------
# Projection: tone curves
# ---------------------------------------------------------------------------

def test_zero_input_gives_black():
    proj, _ = _make_projector()
    assert np.allclose(proj.project(np.zeros((3, 100))), 0.)


def test_output_in_unit_range():
    proj, _ = _make_projector()
    rng = np.random.default_rng(0)
    data = rng.uniform(0., 1., (10, 100))
    rgb = proj.project(data)
    assert rgb.shape == (10, 3)
    assert np.all(rgb >= 0.) and np.all(rgb <= 1.)
    assert np.all(np.isfinite(rgb))


def test_flux_convention_equivalence():
    proj_int, bw = _make_projector(bin_width=0.5)
    proj_den, _ = _make_projector(bin_width=0.5, flux_convention='flux_density')
    rng = np.random.default_rng(1)
    data = rng.uniform(0., 1., (5, 100))
    white = proj_int.luminance_of_spectrum(np.ones(100))
    proj_int.set_luminance_range(Y_saturation=white)
    proj_den.set_luminance_range(Y_saturation=white)
    assert np.allclose(proj_int.project(data), proj_den.project(data/bw), atol=1e-12)


def test_linear_curve_maps_black_and_white_points():
    proj, _ = _make_projector(n_bins=10)
    flat = np.ones(10)
    Y = proj.luminance_of_spectrum(flat)
    proj.set_luminance_range(Y_saturation=Y, Y_black=0.5*Y)
    image = np.stack([flat, 0.5*flat, 0.25*flat])   # at saturation, at black, below black
    rgb = proj.project(image)
    assert rgb[0].max() > 0.9, f"saturation-luminance pixel too dark: {rgb[0]}"
    assert np.allclose(rgb[1], 0., atol=1e-10), f"black-point pixel not black: {rgb[1]}"
    assert np.allclose(rgb[2], 0., atol=1e-10), "below the black point must clamp to black"


def test_black_point_applies_without_log_compression():
    proj, _ = _make_projector(n_bins=10)
    flat = np.ones(10)
    Y = proj.luminance_of_spectrum(flat)
    image = (0.5*flat)[np.newaxis, :]

    proj.set_luminance_range(Y_saturation=Y)
    without_black = proj.project(image)
    proj.set_luminance_range(Y_saturation=Y, Y_black=0.4*Y)
    with_black = proj.project(image)
    assert np.all(with_black <= without_black + 1e-12)
    assert np.any(with_black < without_black - 1e-3), \
        "a linear black point must darken pixels near it"


def test_log_compression_requires_black_point():
    proj, _ = _make_projector(n_bins=10)
    proj.use_log_compression()
    proj.set_luminance_range(Y_saturation=1.)
    with pytest.raises(ValueError, match="black point"):
        proj.project(np.ones((1, 10)))


def test_log_curve_bright_vs_faint():
    proj, _ = _make_projector()
    dr = 1000.
    flat = np.ones(100)
    white = proj.luminance_of_spectrum(flat)
    proj.set_luminance_range(Y_saturation=white, dynamic_range=dr)
    proj.use_log_compression()
    rgb = proj.project(np.stack([flat, flat/dr]))
    assert rgb[0].max() > 0.8, f"bright pixel too dark: {rgb[0]}"
    assert np.all(rgb[1] < 0.02), f"faint pixel not black: {rgb[1]}"


def test_log_curve_midpoint_is_geometric_mean():
    proj, _ = _make_projector(n_bins=10)
    flat = np.ones(10)
    white = proj.luminance_of_spectrum(flat)
    proj.set_luminance_range(Y_saturation=white, dynamic_range=100.)
    proj.use_log_compression()
    captured = {}
    # geometric mean of the range sits exactly halfway up the log curve, and the
    # tone curve leaves the mapped luminance equal to L itself
    proj.project((0.1*flat)[np.newaxis, :],
                 XYZ_inspect_callback=lambda raw, mapped: captured.update(m=mapped))
    assert np.isclose(captured['m'][0, 1], 0.5, rtol=1e-10)


def test_use_log_compression_can_be_switched_off():
    proj, _ = _make_projector(n_bins=10)
    flat = np.ones(10)
    white = proj.luminance_of_spectrum(flat)
    proj.set_luminance_range(Y_saturation=white, dynamic_range=10.)
    proj.use_log_compression()
    log_rgb = proj.project((0.3*flat)[np.newaxis, :])
    proj.use_log_compression(False)
    lin_rgb = proj.project((0.3*flat)[np.newaxis, :])
    assert not np.allclose(log_rgb, lin_rgb)


# ---------------------------------------------------------------------------
# Projection: highlights
# ---------------------------------------------------------------------------

def test_highlights_clamp_preserves_chromaticity_above_saturation():
    proj, _ = _make_projector(n_bins=10)
    spectrum = np.zeros(10)
    spectrum[2] = 1.   # strongly coloured, i.e. far from the sRGB gamut centre
    white = proj.luminance_of_spectrum(spectrum)
    proj.set_luminance_range(Y_saturation=white, highlights='clamp')
    at_white = proj.project(spectrum[np.newaxis, :])
    way_above = proj.project(5.*spectrum[np.newaxis, :])
    assert np.allclose(at_white, way_above), \
        "clamping must render everything above the saturation luminance identically"


def test_highlights_clip_channels_shifts_towards_white():
    proj, _ = _make_projector(n_bins=10)
    spectrum = np.zeros(10)
    spectrum[2] = 1.
    white = proj.luminance_of_spectrum(spectrum)
    proj.set_luminance_range(Y_saturation=white, highlights='clip_channels')
    at_white = proj.project(spectrum[np.newaxis, :])
    way_above = proj.project(5.*spectrum[np.newaxis, :])
    assert not np.allclose(at_white, way_above), \
        "clip_channels must keep changing above the saturation luminance"
    # channels that have not clipped yet keep growing, so the hue drifts; channels
    # already at 0 or 1 stay put, which is exactly the uneven clipping we want
    assert np.all(way_above >= at_white - 1e-12) and np.any(way_above > at_white)


def test_highlights_agree_below_saturation():
    proj, _ = _make_projector(n_bins=10)
    flat = np.ones(10)
    white = proj.luminance_of_spectrum(flat)
    image = (0.5*flat)[np.newaxis, :]
    proj.set_luminance_range(Y_saturation=white, highlights='clamp')
    clamped = proj.project(image)
    proj.set_luminance_range(Y_saturation=white, highlights='clip_channels')
    clipped = proj.project(image)
    assert np.allclose(clamped, clipped)


# ---------------------------------------------------------------------------
# Projection: auto white point
# ---------------------------------------------------------------------------

def test_auto_saturation_luminance_normalises_each_image_separately():
    proj, _ = _make_projector(n_bins=10)
    flat = np.ones((1, 10))
    assert np.allclose(proj.project(flat), proj.project(0.5*flat), atol=1e-10), \
        "without a fixed range each image is normalised to its own maximum"


def test_auto_saturation_luminance_warns_exactly_once():
    proj, _ = _make_projector(n_bins=10)
    handler = _capture_nifty_logs()
    try:
        for _ in range(3):
            proj.project(np.ones((1, 10)))
    finally:
        logger.removeHandler(handler)
    warnings = [r for r in handler.records if r.levelno == logging.WARNING
                and "No luminance range set" in r.getMessage()]
    assert len(warnings) == 1


def test_explicit_range_does_not_warn():
    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=proj.luminance_of_spectrum(np.ones(10)))
    handler = _capture_nifty_logs()
    try:
        proj.project(np.ones((1, 10)))
    finally:
        logger.removeHandler(handler)
    assert not [r for r in handler.records if r.levelno == logging.WARNING]


def test_explicit_range_makes_images_comparable():
    proj, _ = _make_projector(n_bins=10)
    flat = np.ones((1, 10))
    proj.set_luminance_range(Y_saturation=proj.luminance_of_spectrum(np.ones(10)))
    assert np.all(proj.project(0.5*flat) < proj.project(flat))


# ---------------------------------------------------------------------------
# Colour map
# ---------------------------------------------------------------------------

def test_color_map_shape_and_range():
    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=proj.luminance_of_spectrum(np.ones(10)))
    img = proj.get_color_map_image(levels=np.linspace(0., 1., 32))
    assert img.shape == (32, 10, 3)
    assert np.all(img >= 0.) and np.all(img <= 1.)


def test_color_map_uses_the_curve_of_the_last_projection():
    proj, _ = _make_projector(n_bins=10)
    flat = np.ones(10)
    white = proj.luminance_of_spectrum(flat)
    levels = np.geomspace(1e-3, 1., 16)

    proj.set_luminance_range(Y_saturation=white, dynamic_range=100.)
    proj.project(flat[np.newaxis, :])
    linear_map = proj.get_color_map_image(levels=levels)

    proj.use_log_compression()
    proj.project(flat[np.newaxis, :])
    log_map = proj.get_color_map_image(levels=levels)

    assert not np.allclose(linear_map, log_map), \
        "the legend must follow the tone curve the image was rendered with"
    # the log curve is concave, so it sits above the linear one everywhere between
    # the black and white points
    assert log_map.sum() > linear_map.sum(), \
        "log compression lifts intermediate levels relative to linear"


def test_color_map_columns_are_not_rescaled():
    # Bins differ by up to ~10x in luminance per unit flux; the legend must show that.
    proj, _ = _make_projector(n_bins=16)
    proj.set_luminance_range(Y_saturation=proj.luminance_of_spectrum(np.ones(16)))
    proj.project(np.ones((1, 16)))
    img = proj.get_color_map_image(levels=np.array([1.]))
    brightness = img[0].max(axis=-1)
    # the luminance response per unit flux spans ~10x across the visible range; sRGB
    # gamma compresses that, but a per-column rescaling would flatten it entirely
    assert brightness.max() > 1.5*brightness.min(), \
        "columns must keep their genuine relative brightness"


def test_color_map_does_not_disturb_projection_state():
    proj, _ = _make_projector(n_bins=10)
    proj.project(np.ones((1, 10)))
    before = proj._last_transform
    proj.get_color_map_image(levels=np.array([1e6]))
    assert proj._last_transform == before


def test_color_map_without_curve_raises():
    proj, _ = _make_projector(n_bins=10)
    with pytest.raises(RuntimeError):
        proj.get_color_map_image(levels=np.array([1.]))


def test_color_map_default_levels_require_a_projection():
    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=1.)
    with pytest.raises(RuntimeError):
        proj.get_color_map_image()


def test_color_map_default_levels_span_projected_flux_range_and_warn():
    proj, _ = _make_projector(n_bins=10)
    data = np.linspace(0.5, 7., 40)[:, np.newaxis]*np.ones(10)[np.newaxis, :]
    proj.set_luminance_range(Y_saturation=proj.luminance_of_spectrum(np.ones(10)))
    proj.project(data)
    handler = _capture_nifty_logs()
    try:
        proj.get_color_map_image(n_levels=16)
    finally:
        logger.removeHandler(handler)
    assert any("levels default" in r.getMessage() for r in handler.records)
    levels = proj._last_color_map_levels
    assert len(levels) == 16
    assert np.isclose(levels[0], 0.5) and np.isclose(levels[-1], 7.)


# ---------------------------------------------------------------------------
# Colour map ticks
# ---------------------------------------------------------------------------

def test_ticks_require_a_colour_map():
    proj, _ = _make_projector(n_bins=10)
    with pytest.raises(RuntimeError):
        proj.get_color_map_ticks()


def test_ticks_are_round_numbers_inside_the_level_range():
    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=1.)
    levels = np.geomspace(1e-3, 1e2, 128)
    proj.get_color_map_image(levels=levels)
    positions, labels = proj.get_color_map_ticks(n_ticks=5)
    values = [float(lbl) for lbl in labels]
    assert all(levels[0] <= v <= levels[-1] for v in values)
    assert all(np.isclose(np.log10(v) % 1., 0.) or np.isclose(v/10.**np.floor(np.log10(v)),
                                                              round(v/10.**np.floor(np.log10(v))))
               for v in values)
    assert np.all(np.diff(positions) > 0)
    assert positions[0] >= 0. and positions[-1] <= len(levels) - 1


def test_ticks_respect_the_extent():
    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=1.)
    levels = np.linspace(0., 10., 101)
    proj.get_color_map_image(levels=levels)
    pos_idx, labels_idx = proj.get_color_map_ticks(n_ticks=5)
    pos_ext, labels_ext = proj.get_color_map_ticks(n_ticks=5, extent=(0., 1.))
    assert labels_idx == labels_ext
    assert np.allclose(pos_ext, pos_idx/(len(levels) - 1))


def test_apply_color_map_ticks_sets_matplotlib_ticks():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj, _ = _make_projector(n_bins=10)
    proj.set_luminance_range(Y_saturation=1.)
    levels = np.geomspace(1e-2, 1e2, 64)
    img = proj.get_color_map_image(levels=levels)

    fig, ax = plt.subplots()
    ax.imshow(img, origin='lower', aspect='auto', extent=[0.5, 10.5, 0., 1.])
    positions, labels = proj.apply_color_map_ticks(ax, axis='y', n_ticks=5)
    assert np.allclose(ax.get_yticks(), positions)
    assert [t.get_text() for t in ax.get_yticklabels()] == labels
    assert np.all(positions >= 0.) and np.all(positions <= 1.), \
        "positions must land inside the extent the image was drawn with"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visible-bin mapping
# ---------------------------------------------------------------------------

def test_energy_proportional_reflects_input_widths():
    # geomspace bins: each is wider than the previous by a constant ratio
    boundaries = np.geomspace(1., 1000., 11)
    proj = SpectrumToRGBProjector('bin_integrated_flux', 'energy', 'proportional')
    proj.specify_input_spectrum_bins_via_bin_boundaries(boundaries[:-1], boundaries[1:])
    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    vis_widths = lam_upper - lam_lower
    input_widths = boundaries[1:] - boundaries[:-1]
    # visible widths should be proportional to input widths
    ratio = vis_widths / input_widths
    assert np.allclose(ratio, ratio[0], rtol=1e-10), \
        "energy+proportional: visible widths should be proportional to input widths"
    # direction still inverting: higher-index (higher-energy) bins → shorter wavelengths
    assert np.all(np.diff(lam_lower) < 0)


def test_wavelength_uniform_gives_equal_bin_widths():
    proj, _ = _make_projector(spectral_axis_type='wavelength', visible_bin_width='uniform')
    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    vis_widths = lam_upper - lam_lower
    assert np.allclose(vis_widths, vis_widths[0], rtol=1e-10), \
        "wavelength+uniform: all bins should have equal visible Δλ"
    # direction still preserving: higher-index bins → longer wavelengths
    assert np.all(np.diff(lam_lower) > 0)


def test_energy_mode_gives_uniform_bin_widths():
    # 'energy' mode distributes bins equally in visible wavelength regardless of input spacing
    proj, _ = _make_projector()
    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    bin_widths = lam_upper - lam_lower  # always positive with new convention
    assert np.allclose(bin_widths, bin_widths[0], rtol=1e-10), \
        "energy mode should produce uniform Δλ per bin"


def test_energy_mode_direction_inverting():
    # higher input energy → shorter visible wavelength (blue end)
    proj, _ = _make_projector()
    lam_lower, _ = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    assert np.all(np.diff(lam_lower) < 0), \
        "energy mode: higher-index (higher-energy) bins should have shorter visible wavelengths"


def test_wavelength_mode_direction_preserving():
    # higher input wavelength → longer visible wavelength (red end)
    proj, _ = _make_projector(spectral_axis_type='wavelength',
                              visible_bin_width='proportional')
    lam_lower, _ = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    assert np.all(np.diff(lam_lower) > 0), \
        "wavelength mode: higher-index (longer-wavelength) bins should have longer visible wavelengths"


def test_energy_mode_uniform_for_geomspace_input():
    # the key use case: geomspace energy bins should get equal visible Δλ
    boundaries = np.geomspace(1., 1000., 11)
    proj = SpectrumToRGBProjector('bin_integrated_flux', 'energy', 'uniform')
    proj.specify_input_spectrum_bins_via_bin_boundaries(boundaries[:-1], boundaries[1:])
    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    bin_widths = lam_upper - lam_lower
    assert np.allclose(bin_widths, bin_widths[0], rtol=1e-10), \
        "energy mode should give equal visible Δλ even for geomspace input"


def test_d65_illuminant_projects_to_white():
    """The CIE D65 illuminant (sRGB reference white) should project to near-white sRGB.

    This requires using the full CIE 1931 wavelength range (380–780 nm) so that
    the Z tristimulus contributions below 440 nm are included and the projected
    XYZ matches the D65 white point (0.9505, 1.0, 1.089).
    """
    proj = SpectrumToRGBProjector('bin_integrated_flux', 'energy', 'uniform',
                                  wavelength_min_mappable=380.,
                                  wavelength_max_mappable=780.)
    bin_width = 1.0
    centers = bin_width * (0.5 + np.arange(100))
    proj.specify_input_spectrum_bins_via_center_and_width(centers, np.full(100, bin_width))

    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()

    within_bin_wavelengths = np.array(
        [np.linspace(lo, hi, 100) for lo, hi in zip(lam_lower, lam_upper)])
    d65_power = ColorSpaceTools.get_cie_d65_standard_illuminant_power(within_bin_wavelengths)
    # Weight by bin wavelength width so that flux[k] ∝ ∫ D65 dλ over each visible bin.
    # With the default 'energy' mode the bins are uniform in λ (4 nm each), so
    # bin_widths_nm is a constant and drops out after normalisation — but the
    # weighting is written explicitly for correctness in the general case.
    bin_widths_nm = lam_upper - lam_lower
    d65_bin_flux = np.sum(d65_power, axis=1) * bin_widths_nm
    d65_bin_flux /= np.sum(d65_bin_flux)

    srgb = proj.project(d65_bin_flux[np.newaxis, :])
    assert np.allclose(srgb, 1., atol=0.05), f"D65 should project to white, got {srgb}"
