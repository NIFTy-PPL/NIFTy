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

import numpy as np
import pytest

from nifty.cl.spectrum_to_rgb import ColorSpaceTools, SpectrumToRGBProjector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_projector(n_bins=100, bin_width=1.0):
    centers = bin_width * (0.5 + np.arange(n_bins))
    widths = np.full(n_bins, bin_width)
    proj = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform')
    proj.specify_input_spectrum_bins_via_center_and_width(centers, widths)
    return proj, bin_width


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
# SpectrumToRGBProjector — setup / validation
# ---------------------------------------------------------------------------

def test_bin_boundary_spec_rejects_overlap():
    proj = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform')
    lower = np.array([0., 1., 1.5])
    upper = np.array([1.2, 2., 2.5])  # bins 0 and 1 overlap
    with pytest.raises(ValueError):
        proj.specify_input_spectrum_bins_via_bin_boundaries(lower, upper)


def test_bin_spec_rejects_mismatched_lengths():
    proj = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform')
    with pytest.raises(ValueError):
        proj.specify_input_spectrum_bins_via_bin_boundaries(
            np.array([0., 1.]), np.array([1., 2., 3.]))


def test_projection_rejects_negative_flux():
    proj, _ = _make_projector()
    bad = -np.ones((1, 100))
    with pytest.raises(ValueError):
        proj.project_total_spectral_bin_flux(bad)


def test_projection_rejects_wrong_n_bins():
    proj, _ = _make_projector(n_bins=100)
    wrong = np.ones((1, 50))
    with pytest.raises(ValueError):
        proj.project_total_spectral_bin_flux(wrong)


# ---------------------------------------------------------------------------
# SpectrumToRGBProjector — projection correctness
# ---------------------------------------------------------------------------

def test_zero_input_gives_black():
    proj, _ = _make_projector()
    rgb = proj.project_total_spectral_bin_flux(np.zeros((3, 100)))
    assert np.allclose(rgb, 0.)


def test_output_in_unit_range():
    proj, _ = _make_projector()
    rng = np.random.default_rng(0)
    data = rng.uniform(0., 1., (10, 100))
    rgb = proj.project_total_spectral_bin_flux(data)
    assert rgb.shape == (10, 3)
    assert np.all(rgb >= 0.) and np.all(rgb <= 1.)
    assert np.all(np.isfinite(rgb))


def test_flux_convention_equivalence():
    proj, bin_width = _make_projector(bin_width=0.5)
    rng = np.random.default_rng(1)
    data = rng.uniform(0., 1., (5, 100))
    rgb_total = proj.project_total_spectral_bin_flux(data)
    rgb_density = proj.project_spectral_flux_density(data / bin_width)
    assert np.allclose(rgb_total, rgb_density, atol=1e-12)


def test_log_tone_mapping_bright_vs_faint():
    proj, _ = _make_projector()
    dr = 1000.
    bright_row = np.ones((1, 100))
    faint_row = bright_row / dr
    image = np.concatenate([bright_row, faint_row], axis=0)
    rgb = proj.project_total_spectral_bin_flux(image, dynamic_range=dr)
    # bright pixel is at the saturation luminance -> should be near full brightness
    assert rgb[0].max() > 0.8, f"bright pixel too dark: {rgb[0]}"
    # faint pixel is at exactly 1/dynamic_range of the bright one -> black
    assert np.all(rgb[1] < 0.02), f"faint pixel not black: {rgb[1]}"


def test_set_saturation_flux_anchors_absolute_brightness():
    # With auto-scaling, a spectrum and its 2× brighter version look identical
    # (each is normalised to its own max). With a fixed saturation flux set to the
    # brighter level, the dimmer input produces a visibly darker output.
    n_bins, bw = 100, 1.0
    centers = bw * (0.5 + np.arange(n_bins))
    widths = np.full(n_bins, bw)

    def make_proj(saturation_flux=None):
        p = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform')
        p.specify_input_spectrum_bins_via_center_and_width(centers, widths)
        if saturation_flux is not None:
            p.set_saturation_flux(saturation_flux)
        return p

    bright = np.ones((1, n_bins))
    dim    = bright * 0.5

    # Auto-scaling: both flat spectra normalise to their own max → identical sRGB
    assert np.allclose(
        make_proj().project_total_spectral_bin_flux(bright),
        make_proj().project_total_spectral_bin_flux(dim),
        atol=1e-10), "auto-scaling should give identical output for proportionally scaled flat inputs"

    # Fixed saturation anchored to the bright level: dim is darker in every channel
    proj_fixed = make_proj(saturation_flux=float(n_bins * bw))
    rgb_bright = proj_fixed.project_total_spectral_bin_flux(bright)
    rgb_dim    = proj_fixed.project_total_spectral_bin_flux(dim)
    assert np.all(rgb_dim < rgb_bright), \
        "fixed saturation should render the dim input darker than the bright one"


def test_visible_bin_width_invalid_raises():
    with pytest.raises(ValueError):
        SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='invalid')


def test_energy_proportional_reflects_input_widths():
    # geomspace bins: each is wider than the previous by a constant ratio
    boundaries = np.geomspace(1., 1000., 11)
    proj = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='proportional')
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
    proj = SpectrumToRGBProjector(spectral_axis_type='wavelength', visible_bin_width='uniform')
    proj.specify_input_spectrum_bins_via_center_and_width(
        np.arange(100) + 0.5, np.ones(100))
    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    vis_widths = lam_upper - lam_lower
    assert np.allclose(vis_widths, vis_widths[0], rtol=1e-10), \
        "wavelength+uniform: all bins should have equal visible Δλ"
    # direction still preserving: higher-index bins → longer wavelengths
    assert np.all(np.diff(lam_lower) > 0)


def test_spectral_axis_invalid_raises():
    with pytest.raises(ValueError):
        SpectrumToRGBProjector(spectral_axis_type='invalid', visible_bin_width='uniform')


def test_energy_mode_gives_uniform_bin_widths():
    # 'energy' mode distributes bins equally in visible wavelength regardless of input spacing
    proj = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform')
    proj.specify_input_spectrum_bins_via_center_and_width(
        np.arange(100) + 0.5, np.ones(100))
    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    bin_widths = lam_upper - lam_lower  # always positive with new convention
    assert np.allclose(bin_widths, bin_widths[0], rtol=1e-10), \
        "energy mode should produce uniform Δλ per bin"


def test_energy_mode_direction_inverting():
    # higher input energy → shorter visible wavelength (blue end)
    proj = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform')
    proj.specify_input_spectrum_bins_via_center_and_width(
        np.arange(100) + 0.5, np.ones(100))
    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    assert np.all(np.diff(lam_lower) < 0), \
        "energy mode: higher-index (higher-energy) bins should have shorter visible wavelengths"


def test_wavelength_mode_direction_preserving():
    # higher input wavelength → longer visible wavelength (red end)
    proj = SpectrumToRGBProjector(spectral_axis_type='wavelength', visible_bin_width='proportional')
    proj.specify_input_spectrum_bins_via_center_and_width(
        np.arange(100) + 0.5, np.ones(100))
    lam_lower, lam_upper = proj._map_input_spectrum_bins_to_visible_light_wavelength_bins()
    assert np.all(np.diff(lam_lower) > 0), \
        "wavelength mode: higher-index (longer-wavelength) bins should have longer visible wavelengths"


def test_energy_mode_uniform_for_geomspace_input():
    # the key use case: geomspace energy bins should get equal visible Δλ
    boundaries = np.geomspace(1., 1000., 11)
    proj = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform')
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
    proj = SpectrumToRGBProjector(spectral_axis_type='energy', visible_bin_width='uniform',
                                   wavelength_min_mappable=380., wavelength_max_mappable=780.)
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

    srgb = proj.project_total_spectral_bin_flux(d65_bin_flux[np.newaxis, :])
    assert np.allclose(srgb, 1., atol=0.05), f"D65 should project to white, got {srgb}"
