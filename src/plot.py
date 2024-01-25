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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import os
from datetime import datetime as dt
from itertools import product
from warnings import warn

import numpy as np

from .domain_tuple import DomainTuple
from .domains.gl_space import GLSpace
from .domains.hp_space import HPSpace
from .domains.power_space import PowerSpace
from .domains.rg_space import RGSpace
from .field import Field
from .minimization.iteration_controllers import EnergyHistory
from .multi_field import MultiField
from .utilities import check_object_identity, myassert

# relevant properties:
# - x/y size
# - x/y/z log
# - x/y/z min/max
# - colorbar/colormap
# - axis on/off
# - title
# - axis labels
# - labels


def _mollweide_helper(xsize):
    xsize = int(xsize)
    ysize = xsize // 2
    res = np.full(shape=(ysize, xsize), fill_value=np.nan, dtype=np.float64)
    xc, yc = (xsize - 1) * 0.5, (ysize - 1) * 0.5
    u, v = np.meshgrid(np.arange(xsize), np.arange(ysize))
    u, v = 2 * (u - xc) / (xc / 1.02), (v - yc) / (yc / 1.02)

    mask = np.where((u * u * 0.25 + v * v) <= 1.)
    t1 = v[mask]
    theta = 0.5 * np.pi - (np.arcsin(2 / np.pi * (np.arcsin(t1) + t1 * np.sqrt(
        (1. - t1) * (1 + t1)))))
    phi = -0.5 * np.pi * u[mask] / np.maximum(np.sqrt((1 - t1) * (1 + t1)), 1e-6)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return res, mask, theta, phi


def _hammer_helper(xsize):
    xsize = int(xsize)
    ysize = xsize // 2
    res = np.full(shape=(ysize, xsize), fill_value=np.nan, dtype=np.float64)
    xc, yc = (xsize - 1) * 0.5, (ysize - 1) * 0.5
    u, v = np.meshgrid(np.arange(xsize), np.arange(ysize))
    u, v = 2 * (u - xc) / (xc / 1.02), (v - yc) / (yc / 1.02)

    u *= np.sqrt(2)
    v *= np.sqrt(2)

    mask = np.where((u * u / 8 + v * v / 2) <= 1.)
    umask, vmask = u[mask], v[mask]

    zmask = np.sqrt(1 - umask * umask / 16 - vmask * vmask / 4)
    longitude = 2 * np.arctan(zmask * umask / 2 / (2 * zmask * zmask - 1.))
    latitude = np.arcsin(zmask * vmask)

    theta = np.pi / 2 - latitude
    phi = -longitude

    assert np.min(theta) >= 0.
    assert np.max(theta) <= np.pi

    return res, mask, theta, phi


_sphere_projection_helpers = {
    'mollweide': _mollweide_helper,
    'hammer': _hammer_helper,
}


def project_spherical_data_to_2d(val, domain, projection='mollweide', xsize=800, have_rgb=False):
    """Projects spherical data `val` onto a two-dimensional plane.

    Parameters
    ----------
    val : :class:`numpy.ndarray`
        Values of the spherical field to be projected.
    domain : :class:`~nifty8.domains.hp_space.HPSpace`, :class:`~nifty8.domains.gl_space.GLSpace`
        Domain of the spherical field.
    projection : string
        Name of the projection to be used. Currently supported: 'mollweide', 'hammer'.
        Default: 'mollweide'.
    xsize : int, float
        Number of pixels of the output image in the horizontal dimension.
        ysize is chosen automatically.
    have_rgb : boolean
        Whether `val` contains RGB values (in an extra dimension).
        Needed for multi-frequency plotting.

    Returns
    -------
    res : :class:`numpy.ndarray`
        The image resulting from the projection.
    """
    if projection not in _sphere_projection_helpers.keys():
        raise ValueError(f"Projection '{projection}' unsupported!")

    res, mask, theta, phi = _sphere_projection_helpers[projection](xsize)

    if have_rgb:
        res = np.full(shape=res.shape + (3, ), fill_value=1., dtype=np.float64)

    if isinstance(domain, HPSpace):
        from ducc0.healpix import Healpix_Base
        ptg = np.empty((phi.size, 2), dtype=np.float64)
        ptg[:, 0] = theta
        ptg[:, 1] = phi
        base = Healpix_Base(int(np.sqrt(domain.size // 12)), "RING")
        res[mask] = val[base.ang2pix(ptg)]
    else:
        from ducc0.misc import GL_thetas
        ra = np.linspace(0, 2 * np.pi, domain.nlon + 1)
        dec = GL_thetas(domain.nlat)
        ilat = _find_closest(dec, theta)
        ilon = _find_closest(ra, phi)
        ilon = np.where(ilon == domain.nlon, 0, ilon)
        res[mask] = val[ilat * domain.nlon + ilon]

    return res


class SpectrumToRGBProjector:
    """Class to facilitate projections of fields with a spectral dimension to sRGBs color space.

    The fields spectral dimension is mapped onto the visible light spectral range.
    The visible light spectra created this way are then mapped to percieved colors
    following the CIE 1931 model of human color perception. The percieved colors
    are then encoded into the sRGB color space and an array of sRGB values is returned.

    The code makes a distinction between fluxes :math:`\Phi` and spectral
    flux densities :math:`\del\Phi/\del\E` and provides separate projection
    functions for each: :func:`project_integrated_flux` and
    :func:`project_flux_density`.

    To set up the class for projections, the spectral bin boundaries need to be specified.
    The class provides two methods for this purpose:
    :func:`specify_input_spectrum_bins_via_bin_boundaries` allows communicating
    both linear and non-linear energy binning schemes by specifying the upper and lower
    boundaries of the bins.
    For linearly spaced energy bins there additionally is the convenience function
    :func:`specify_input_spectrum_bins_via_center_energy_and_width`.

    Second, if desireable, a fixed white point can be specified. 
    This is possible either via a total spectrum-wide flux at which
    color saturation should occur or via a flux density at which it should occur,
    in combination with a specification of how densely/sparsly populated the average
    plotted spectrum is assumed to be.
    For this, see the methods :func:`set_saturation_flux` and
    :func:`set_saturation_flux_density`.
    If no white point is set via these methods, it will be chosen for each input
    automatically such that the brightest pixel defines the upper saturation point.

    Parameters
    ----------
    wavelength_min_mappable : int, float
        Short wavelength limit (nm) of the visible light spectrum mapped to.
        Affects what color the highest energy bin is mapped to.
        Default: 440 nm
    wavelength_max_mappable : int, float
        Long wavelength limit (nm) of the visible light spectrum mapped to.
        Affects what color the lowest energy bin is mapped to.
        Default: 640 nm
    """
    _input_spectrum_bin_lower_energies = None
    _input_spectrum_bin_upper_energies = None
    _input_spectrum_bin_widths = None
    _input_spectrum_relative_bin_widths = None
    _input_spectrum_bin_distortion_fn = None

    _input_saturation_flux = None

    _visible_spectrum_bin_lower_wavelengths = None
    _visible_spectrum_bin_upper_wavelengths = None
    _visible_spectrum_bin_widths = None

    _visible_spectrum_bin_flux_to_XYZ_mapping_tensor = None

    # --- init function ---
    def __init__(self, wavelength_min_mappable=440., wavelength_max_mappable=640.):
        self._WAVELENGTH_MIN_MAPPABLE = self._check_pos_scalar(wavelength_min_mappable, "wavelength_min_mappable")
        self._WAVELENGTH_MAX_MAPPABLE = self._check_pos_scalar(wavelength_max_mappable, "wavelength_max_mappable")

    # --- Functions to provide input energy bin layout ---
    def specify_input_spectrum_bins_via_bin_boundaries(self, lower_energies, upper_energies):
        if len(lower_energies) != len(upper_energies):
            raise ValueError("equal number of upper and lower bin boundaries needs to be given")
        widths = upper_energies - lower_energies
        if np.any(widths <= 0.):
            raise ValueError("bin width cannot be zero or negative")
        sort_idx_lower = np.argsort(lower_energies)
        sort_idx_upper = np.argsort(upper_energies)
        if not np.all(sort_idx_lower == sort_idx_upper):
            raise ValueError("either bins overlap or the bins are not sorted the same in both arrays.")
        if np.any(upper_energies[sort_idx_upper][:-1] > lower_energies[sort_idx_lower][1:]):
            raise ValueError("spectral bins cannot overlap")

        self._input_spectrum_bin_lower_energies = lower_energies
        self._input_spectrum_bin_upper_energies = upper_energies

        if self._input_spectrum_bin_widths is None:
            self._input_spectrum_bin_widths = widths
            self._input_spectrum_relative_bin_widths = widths / np.sum(widths)
        else:
            if not np.allclose(widths, self._input_spectrum_bin_widths):
                raise ValueError("inconsistent definition of bins - " +
                                 "did you call multiple bin specification functions?")

    def specify_input_spectrum_bins_via_center_energy_and_width(self, center_energies, widths):
        if np.any(widths <= 0.):
            raise ValueError("Spectral bins widths need to be positive.")
        self._input_spectrum_bin_widths = widths
        self._input_spectrum_relative_bin_widths = widths / np.sum(widths)

        lower_energies = center_energies - widths / 2.0
        upper_energies = center_energies + widths / 2.0
        self.specify_input_spectrum_bins_via_bin_boundaries(lower_energies, upper_energies)

    # --- Functions to influence the satuartion point ---
    def set_saturation_flux(self, saturation_flux):
        """Sets the total flux by which the upper saturation point will be defined.

        Parameters
        ----------
        saturation_flux : float
            Total (spectrally integrated) flux which should define the upper saturation point.
        """
        self._input_saturation_flux = float(self._check_pos_scalar(saturation_flux, "saturation flux"))

    def set_saturation_flux_density(self, saturation_flux_density, spectral_denseness):
        """Sets the flux density and spectral denseness by which the upper saturation point will be defined.

        As brightness perception is approximately linear in total light flux, but spectra
        can vary from sparse (monochromatic source) to dense (flat-spectrum source),
        we need to specify over which spectral range we need to integrate the given flux density
        to estimate a reasonable upper saturation point in terms of total flux for the given application.

        We specify this in the form of a `spectral_denseness`, which should be set close to `1.0`
        for flat spectra and close to `1 / peak width` or `1 / bin width` for strongly peaked spectra.

        This method requires that spectral bin widths have been specified in the initialization
        of the class.

        Parameters
        ----------
        saturation_flux_densty : float
            Flux density which should define the upper saturation point.
        spectral_denseness : float in (0., 1.]
            How densly the processed spectrum fill the spectral domain (~1/spectral sparseness).
        """
        saturation_flux_density = float(
            self._check_pos_scalar(saturation_flux_density, "saturation flux denstity"))
        if not self._is_scalar(spectral_denseness):
            raise ValueError("spectral denseness needs to be a scalar")
        if spectral_denseness <= 0. or spectral_denseness > 1.:
            raise ValueError("spectral denseness needs to be within (0., 1.]")
        if self._input_spectrum_bin_widths is None:
            raise RuntimeError("this method requires the spectral bin widths "
                               "to be specified during initialization.")
        self._input_saturation_flux = saturation_flux_density \
            * np.sum(self._input_spectrum_bin_widths) * spectral_denseness

    # --- functions to implement saturation effect ---
    def _apply_luminance_saturation(self, XYZ_data, reference_bin_flux_spectrum=None):
        """Creates saturation in luminosity.

        For this, it divides the XYZ input values by a reference luminosity,
        projects them into xyZ (chromaticity-luminosity) space, applies a
        clipping on the luminosity values and transforms back to XYZ space.

        If a reference flux spectrum is given, the reference luminosity
        will be derived from it.
        Otherwise, the largest Y value passed is taken as the reference
        luminosity.

        Parameters
        ----------
        XYZ_data : :class:`numpy.ndarray`
            CIE 1931 XYZ tristimulus values to be processesd
        reference_bin_flux_spectrum : :class:`numpy.ndarray`, None
            Reference visible light spectrum for luminosity saturation
            point derivation.
        """
        if reference_bin_flux_spectrum is None:
            saturation_luminance = np.max(XYZ_data[..., 1])
        else:
            saturation_luminance = np.dot(
                reference_bin_flux_spectrum,
                self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor[:, 1])

        xyY_data = ColorSpaceTools.XYZ_to_xyY(XYZ_data / saturation_luminance)
        return ColorSpaceTools.xyY_to_XYZ(xyY_data.clip(0., 1.))

    def _apply_cone_response_saturation(self, XYZ_data, reference_bin_flux_spectrum=None):
        """Simulates saturation in the response of the retinal cone cells.

        For this, it projects the XYZ input values into the LMS (cone response) space,
        divides by LMS reference values and clips the result to [0., 1.].
        The clipped LMS values are transformed back to XYZ values and returned.

        If a reference flux spectrum is given, the LMS reference values
        are derived from it.
        Otherwise, the largest value passed will be taken as the reference
        values.

        Parameters
        ----------
        XYZ_data : :class:`numpy.ndarray`
            CIE 1931 XYZ tristimulus values to be processesd
        reference_bin_flux_spectrum : :class:`numpy.ndarray`, None
            Reference visible light spectrum for L, M, and S saturation
            point derivation.
        """
        LMS_data = ColorSpaceTools.XYZ_to_LMS(XYZ_data)

        if reference_bin_flux_spectrum is None:
            saturation_LMS = np.array([np.max(LMS_data[..., i]) for i in range(3)])
            print(saturation_LMS)  #DEBUG
        else:
            reference_bin_flux_spectrum = reference_bin_flux_spectrum[np.newaxis, :]
            saturation_XYZ = self._transform_visible_spectrum_bin_flux_to_XYZ(
                reference_bin_flux_spectrum)
            saturation_LMS = ColorSpaceTools.XYZ_to_LMS(saturation_XYZ)[0]

        broadcast_slice = (None, ) * (LMS_data.ndim - 1) + (slice(None), )
        LMS_data /= saturation_LMS[broadcast_slice]

        return ColorSpaceTools.LMS_to_XYZ(LMS_data.clip(0., 1.))

    # --- function to associate visible light bins with input energy bins
    def map_input_spectrum_bins_to_visible_light_wavelength_bins(self,
                                                                 lower_bin_energies=None,
                                                                 upper_bin_energies=None):
        """Maps given bin energy values into the visible spectrum energy range
        by linearly mapping input energies to visible photon energies.

        All parameters are optional - if `None` is given, the class-internal
        values are used.

        Parameters
        ----------
        lower_bin_energies : :class:`numpy.ndarray`, list of float, None
            Lower energy boundaries of the spectral domain bins.
        upper_bin_energies : :class:`numpy.ndarray`, list of float, None
            Upper energy boundaries of the spectral domain bins.

        Returns
        -------
        lambda_visible_lower : :class:`numpy.ndarray`
            Mapper lower wavelength boundaries of the given spectral bins.
        lambda_visible_upper : :class:`numpy.ndarray`
            Mapper upper wavelength boundaries of the given spectral bins.
        """
        E_in_lower = lower_bin_energies if lower_bin_energies is not None else self._input_spectrum_bin_lower_energies
        E_in_upper = upper_bin_energies if upper_bin_energies is not None else self._input_spectrum_bin_upper_energies

        if np.any(E_in_upper - E_in_lower <= 0):
            raise ValueError("bins of negative width given")

        E_in_min = np.min(E_in_lower)
        E_in_max = np.max(E_in_upper)

        E0_vis, E1_vis = 1. / self._WAVELENGTH_MAX_MAPPABLE, 1. / self._WAVELENGTH_MIN_MAPPABLE

        E_visible_lower = E0_vis + (E_in_lower - E_in_min) / (E_in_max - E_in_min) * (E1_vis - E0_vis)
        E_visible_upper = E0_vis + (E_in_upper - E_in_min) / (E_in_max - E_in_min) * (E1_vis - E0_vis)

        lambda_vis_lower = 1. / E_visible_lower
        lambda_vis_upper = 1. / E_visible_upper

        return lambda_vis_lower, lambda_vis_upper

    # --- functions to map visible light fluxse to XYZ values ---
    def _transform_visible_spectrum_bin_flux_to_XYZ(self, vis_bin_flux):
        return np.tensordot(vis_bin_flux,
                            self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor,
                            axes=(-1, 0))

    def _generate_visible_spectrum_bin_flux_to_XYZ_mapping_tensor(self):
        # Average CIE1931 tristimulus color mapping functions within each visible light bin,
        # assuming constant emissivity within the bins.
        # See https://en.wikipedia.org/wiki/CIE_1931_color_space#Emissive_case

        # get visible light bin boundaries
        lambda_visible_lower, lambda_visible_upper = self.map_input_spectrum_bins_to_visible_light_wavelength_bins()
        self._visible_spectrum_bin_lower_wavelengths = lambda_visible_lower
        self._visible_spectrum_bin_upper_wavelengths = lambda_visible_upper
        self._visible_spectrum_bin_widths = lambda_visible_upper - lambda_visible_lower

        # compute average tristimulus values for wavelengths within visible spectrum bins
        lambda_gen = zip(lambda_visible_lower, lambda_visible_upper)
        within_bin_wavelengths = 1. / np.array(
            [np.linspace(1./wl_lower, 1./wl_upper, 100) for wl_lower, wl_upper in lambda_gen])
        XYZ_values_of_within_bin_wavelengths = ColorSpaceTools.get_cie1931_standard_observer_XYZ_tristimulus_values(within_bin_wavelengths)
        self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor = np.mean(XYZ_values_of_within_bin_wavelengths, axis=1)

    # --- functions to perform the full projection from energy spectrum to sRGB value ---
    def project_spectral_flux_density(self, spectral_flux_density, saturation_via='luminance'):
        """Projects spectral flux density data to percieved colors and embeds in sRGB.

        Parameters
        ----------
        spectral_flux_density : :class:`numpy.ndarray`
            Values of the field to be projected.
            The spectral dimension is expected to be the last dimension of the array.
        saturation_via : string
            Type of saturation to apply. Supported values are `luminance` (default)
            and `retinal cone response`.

        Returns
        -------
        sRGB_data : :class:`numpy.ndarray`
            The transformed image, containing the sRGB values in the last dimension.
        """
        self._pre_projection_checks(spectral_flux_density)
        broadcast_sl = (None, ) * (spectral_flux_density.ndim - 1) + (slice(None), )
        total_spectral_bin_flux = spectral_flux_density * self._input_spectrum_bin_widths[broadcast_sl]
        self.project_total_spectral_bin_flux(total_spectral_bin_flux, saturation_via=saturation_via)

    def project_total_spectral_bin_flux(self, total_spectral_bin_flux, saturation_via='luminance'):
        """Projects spectral bin flux data to percieved colors and embeds in sRGB.

        Parameters
        ----------
        spectral_bin_flux : :class:`numpy.ndarray`
            Values of the field to be projected.
            The spectral dimension is expected to be the last dimension of the array.
        saturation_via : string
            Type of saturation to apply. Supported values are `'luminance'` (default)
            and `'retinal cone response'`.

        Returns
        -------
        sRGB_data : :class:`numpy.ndarray`
            The transformed image, containing the sRGB values in the last dimension.
        """
        self._pre_projection_checks(total_spectral_bin_flux)

        visible_bin_flux = total_spectral_bin_flux

        XYZ_data = self._transform_visible_spectrum_bin_flux_to_XYZ(visible_bin_flux)

        saturation_spectral_bin_flux = None if self._input_saturation_flux is None else \
            self._input_saturation_flux * self._input_spectrum_relative_bin_widths

        if saturation_via == 'luminance':
            saturation_function = self._apply_luminance_saturation
        elif saturation_via == 'retinal cone response':
            saturation_function = self._apply_cone_response_saturation
        else:
            raise ValueError("Unknown saturation function '{saturation_via}'")
        XYZ_data_saturated = saturation_function(XYZ_data, saturation_spectral_bin_flux)

        # embed to sRGB
        sRGB_data = ColorSpaceTools.embed_XYZ_perceived_color_in_sRGB(XYZ_data_saturated)

        return sRGB_data

    def _pre_projection_checks(self, input_data):
        if self._input_spectrum_bin_widths is None:
            raise ValueError("Spectral bins need to be specified before projection")

        if np.any(input_data < 0.):
            raise ValueError("Fluxes and flux densities must be positive or zero")

        if input_data.shape[-1] != self._input_spectrum_bin_lower_energies.shape[0]:
            raise ValueError("Projector initialized with incompatible spectral bins.")

        if self._visible_spectrum_bin_flux_to_XYZ_mapping_tensor is None:
            self._generate_visible_spectrum_bin_flux_to_XYZ_mapping_tensor()

    def _get_d65_bin_flux_for_visible_spectrum_bins(self):
        lambda_visible_lower = self._visible_spectrum_bin_lower_wavelengths
        lambda_visible_upper = self._visible_spectrum_bin_upper_wavelengths
        within_bin_wavelengths = 1. / np.array([
            np.linspace(1. / l_lower, 1. / l_upper, 100)
            for l_lower, l_upper in zip(lambda_visible_lower, lambda_visible_upper)
        ])
        relative_power = ColorSpaceTools.get_cie_d65_standard_illuminant_relative_power(
            within_bin_wavelengths)
        res = np.sum(relative_power, axis=1)
        return res / np.sum(res)

    # --- check functions ---
    def _check_pos_scalar(self, inp, name):
        """Checks wether input is a strictly positive scalar"""
        if not self._is_scalar(inp):
            raise ValueError(name + "needs to be a scalar")
        if inp <= 0.:
            raise ValueError(name + "must me strictly positive")
        return inp

    def _check_override(self, inp, name, default):
        if inp == False:
            return default
        return self._check_pos_scalar(inp, name)

    def _is_scalar(self, inp):
        return self._is_scalar_number(inp) or self._is_scalar_ndarray(inp)

    def _is_scalar_number(self, inp):
        import numbers
        return isinstance(inp, numbers.Number)

    def _is_scalar_ndarray(self, inp):
        return isinstance(inp, np.ndarray) and inp.shape == ()

    # --- preprocessing functions ---
    def to_logscale(self, spectral_values, vmin, vmax):
        res = spectral_values.clip(vmin, vmax)
        res = np.log(res / vmin)  # >= 0. by design
        res /= np.log(vmax / vmin)  # <= 1. by design
        return res

    def after_log_gammacorr(spectral_values_log, exponent):
        return np.float_power(spectral_values_log, exponent)


class ColorSpaceTools:
    # --- Color space trafos ---
    def XYZ_to_xyY(XYZ_values):
        res = XYZ_values / np.sum(XYZ_values, axis=-1)[..., np.newaxis]
        res[..., 2] = XYZ_values[..., 1]
        return res

    def xyY_to_XYZ(xyY_values):
        res = np.empty_like(xyY_values)
        fct = xyY_values[..., 2] / xyY_values[..., 1]
        res[..., 0] = xyY_values[..., 0] * fct
        res[..., 1] = xyY_values[..., 2]
        res[..., 2] = (1. - xyY_values[..., 0] - xyY_values[..., 1]) * fct
        return res

    @classmethod
    def XYZ_to_LMS(self, XYZ_values):
        """Transform CIE1931 XYZ tristimulus values to LMS retinal cone response values."""
        # Stockman & Sharpe 2000
        return np.tensordot(XYZ_values, self._XYZ_to_LMS_mat, axes=(-1, 1))

    _XYZ_to_LMS_mat = np.array([[0.210576, 0.855098, -0.0396983],
                               [-0.417076, 1.177260, 0.0786283],
                               [0.0, 0.0, 0.5168350]])

    @classmethod
    def LMS_to_XYZ(self, LMS_values):
        """Transform LMS retinal cone response values to CIE1931 XYZ tristimulus values."""
        # Stockman & Sharpe 2000
        return np.tensordot(LMS_values, self._LMS_to_XYZ_mat, axes=(-1, 1))

    _LMS_to_XYZ_mat = np.array([[1.94735469, -1.41445123, 0.36476327],
                                [0.68990272, 0.34832189, 0.0],
                                [0.0, 0.0, 1.93485343]])

    @classmethod
    def embed_XYZ_perceived_color_in_sRGB(self, XYZ_values):
        """Transform CIE 1931 XYZ tristimulus values to corresponding sRGB values,
        as best as possible.

        Parameters
        ----------
        XYZ_values : :class:`numpy.ndarray`
            XYZ tristimulus values

        Returns
        -------
        res_sRGB : :class:`numpy.ndarray`
            Corresponding sRGB values.
        """
        tmp = np.tensordot(self._CIE1931_XYZ_TO_sRGB_D65, XYZ_values, axes=(1, -1)).T
        tmp = tmp.clip(0., 1.)  # clip to values inside the sRGB garmut
        return self._sRGB_gammacorr(tmp)

    _CIE1931_XYZ_TO_sRGB_D65 = np.array([[3.2404542, -1.5371385, -0.4985314],
                                         [-0.9692660, 1.8760108, 0.0415560],
                                         [0.0556434, -0.2040259, 1.0572252]])

    def _sRGB_gammacorr(inp):
        """Perform gamma correction according to sRGB standard."""
        mask = np.zeros(inp.shape, dtype=np.float64)
        mask[inp <= 0.0031308] = 1.
        r1 = 12.92 * inp
        a = 0.055
        r2 = (1 + a) * (np.maximum(inp, 0.0031308) ** (1 / 2.4)) - a
        return r1 * mask + r2 * (1. - mask)

    # --- Standardized observer and source models ---
    @classmethod
    def get_cie1931_standard_observer_XYZ_tristimulus_values(self, wavelengths):
        """Get CIE 1931 XYZ tristimulus values for given wavelengths.

        Linearly interpolates in the CIE 1931 Standard Observer table
        (:math:`\delta\lambda = \mathrm{5nm}`).

        Parameters
        ----------
        wavelengths : int, float, :class:`numpy.ndarray`
            Wavelength(s) for which to compute the equivalent raw RGB intensities.

        Returns
        -------
        XYZ_tristimulus_values : :class:`numpy.ndarray`
            CIE 1931 XYZ tristimulus values for the requested wavelength(s).
        """
        res = np.empty(wavelengths.shape + (3, ))
        for i in range(3):
            res[..., i] = np.interp(wavelengths,
                                    self._CIE1931_STANDARD_OBSERVER_WAVELENGTH_TABLE_380nm_TO_780nm,
                                    self._CIE1931_STANDARD_OBSERVER_XYZ_COLOR_MATCHING_TABLE_380nm_TO_780nm[i])
        return res

    _CIE1931_STANDARD_OBSERVER_WAVELENGTH_TABLE_380nm_TO_780nm = np.linspace(380., 780., 81)
    _CIE1931_STANDARD_OBSERVER_XYZ_COLOR_MATCHING_TABLE_380nm_TO_780nm = np.array(
          [[0.000160, 0.000662, 0.002362, 0.007242, 0.019110,
            0.043400, 0.084736, 0.140638, 0.204492, 0.264737,
            0.314679, 0.357719, 0.383734, 0.386726, 0.370702,
            0.342957, 0.302273, 0.254085, 0.195618, 0.132349,
            0.080507, 0.041072, 0.016172, 0.005132, 0.003816,
            0.015444, 0.037465, 0.071358, 0.117749, 0.172953,
            0.236491, 0.304213, 0.376772, 0.451584, 0.529826,
            0.616053, 0.705224, 0.793832, 0.878655, 0.951162,
            1.014160, 1.074300, 1.118520, 1.134300, 1.123990,
            1.089100, 1.030480, 0.950740, 0.856297, 0.754930,
            0.647467, 0.535110, 0.431567, 0.343690, 0.268329,
            0.204300, 0.152568, 0.112210, 0.081261, 0.057930,
            0.040851, 0.028623, 0.019941, 0.013842, 0.009577,
            0.006605, 0.004553, 0.003145, 0.002175, 0.001506,
            0.001045, 0.000727, 0.000508, 0.000356, 0.000251,
            0.000178, 0.000126, 0.000090, 0.000065, 0.000046,
            0.000033],
           [0.000017, 0.000072, 0.000253, 0.000769, 0.002004,
            0.004509, 0.008756, 0.014456, 0.021391, 0.029497,
            0.038676, 0.049602, 0.062077, 0.074704, 0.089456,
            0.106256, 0.128201, 0.152761, 0.185190, 0.219940,
            0.253589, 0.297665, 0.339133, 0.395379, 0.460777,
            0.531360, 0.606741, 0.685660, 0.761757, 0.823330,
            0.875211, 0.923810, 0.961988, 0.982200, 0.991761,
            0.999110, 0.997340, 0.982380, 0.955552, 0.915175,
            0.868934, 0.825623, 0.777405, 0.720353, 0.658341,
            0.593878, 0.527963, 0.461834, 0.398057, 0.339554,
            0.283493, 0.228254, 0.179828, 0.140211, 0.107633,
            0.081187, 0.060281, 0.044096, 0.031800, 0.022602,
            0.015905, 0.011130, 0.007749, 0.005375, 0.003718,
            0.002565, 0.001768, 0.001222, 0.000846, 0.000586,
            0.000407, 0.000284, 0.000199, 0.000140, 0.000098,
            0.000070, 0.000050, 0.000036, 0.000025, 0.000018,
            0.000013],
           [0.000705, 0.002928, 0.010482, 0.032344, 0.086011,
            0.197120, 0.389366, 0.656760, 0.972542, 1.282500,
            1.553480, 1.798500, 1.967280, 2.027300, 1.994800,
            1.900700, 1.745370, 1.554900, 1.317560, 1.030200,
            0.772125, 0.570060, 0.415254, 0.302356, 0.218502,
            0.159249, 0.112044, 0.082248, 0.060709, 0.043050,
            0.030451, 0.020584, 0.013676, 0.007918, 0.003988,
            0.001091, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000]])

    @classmethod
    def get_cie_d65_standard_illuminant_relative_power(self, wavelengths):
        return np.interp(wavelengths,
                         self._CIE_D65_WAVELENGTH_TABLE_300nm_TO_830nm,
                         self._CIE_D65_RELATIVE_POWER_TABLE_300nm_TO_830nm)

    # CIE 2022, CIE standard illumintant D65, DOI: 10.25039/CIE.DS.hjfjmt59
    _CIE_D65_WAVELENGTH_TABLE_300nm_TO_830nm = np.linspace(300., 830., 531)
    _CIE_D65_RELATIVE_POWER_TABLE_300nm_TO_830nm = np.array(
        [3.41000e-02, 3.60140e-01, 6.86180e-01, 1.01222e+00, 1.33826e+00,
         1.66430e+00, 1.99034e+00, 2.31638e+00, 2.64242e+00, 2.96846e+00,
         3.29450e+00, 4.98865e+00, 6.68280e+00, 8.37695e+00, 1.00711e+01,
         1.17652e+01, 1.34594e+01, 1.51535e+01, 1.68477e+01, 1.85418e+01,
         2.02360e+01, 2.19177e+01, 2.35995e+01, 2.52812e+01, 2.69630e+01,
         2.86447e+01, 3.03265e+01, 3.20082e+01, 3.36900e+01, 3.53717e+01,
         3.70535e+01, 3.73430e+01, 3.76326e+01, 3.79221e+01, 3.82116e+01,
         3.85011e+01, 3.87907e+01, 3.90802e+01, 3.93697e+01, 3.96593e+01,
         3.99488e+01, 4.04451e+01, 4.09414e+01, 4.14377e+01, 4.19340e+01,
         4.24302e+01, 4.29265e+01, 4.34228e+01, 4.39191e+01, 4.44154e+01,
         4.49117e+01, 4.50844e+01, 4.52570e+01, 4.54297e+01, 4.56023e+01,
         4.57750e+01, 4.59477e+01, 4.61203e+01, 4.62930e+01, 4.64656e+01,
         4.66383e+01, 4.71834e+01, 4.77285e+01, 4.82735e+01, 4.88186e+01,
         4.93637e+01, 4.99088e+01, 5.04539e+01, 5.09989e+01, 5.15440e+01,
         5.20891e+01, 5.18777e+01, 5.16664e+01, 5.14550e+01, 5.12437e+01,
         5.10323e+01, 5.08209e+01, 5.06096e+01, 5.03982e+01, 5.01869e+01,
         4.99755e+01, 5.04428e+01, 5.09100e+01, 5.13773e+01, 5.18446e+01,
         5.23118e+01, 5.27791e+01, 5.32464e+01, 5.37137e+01, 5.41809e+01,
         5.46482e+01, 5.74589e+01, 6.02695e+01, 6.30802e+01, 6.58909e+01,
         6.87015e+01, 7.15122e+01, 7.43229e+01, 7.71336e+01, 7.99442e+01,
         8.27549e+01, 8.36280e+01, 8.45011e+01, 8.53742e+01, 8.62473e+01,
         8.71204e+01, 8.79936e+01, 8.88667e+01, 8.97398e+01, 9.06129e+01,
         9.14860e+01, 9.16806e+01, 9.18752e+01, 9.20697e+01, 9.22643e+01,
         9.24589e+01, 9.26535e+01, 9.28481e+01, 9.30426e+01, 9.32372e+01,
         9.34318e+01, 9.27568e+01, 9.20819e+01, 9.14069e+01, 9.07320e+01,
         9.00570e+01, 8.93821e+01, 8.87071e+01, 8.80322e+01, 8.73572e+01,
         8.66823e+01, 8.85006e+01, 9.03188e+01, 9.21371e+01, 9.39554e+01,
         9.57736e+01, 9.75919e+01, 9.94102e+01, 1.01228e+02, 1.03047e+02,
         1.04865e+02, 1.06079e+02, 1.07294e+02, 1.08508e+02, 1.09722e+02,
         1.10936e+02, 1.12151e+02, 1.13365e+02, 1.14579e+02, 1.15794e+02,
         1.17008e+02, 1.17088e+02, 1.17169e+02, 1.17249e+02, 1.17330e+02,
         1.17410e+02, 1.17490e+02, 1.17571e+02, 1.17651e+02, 1.17732e+02,
         1.17812e+02, 1.17517e+02, 1.17222e+02, 1.16927e+02, 1.16632e+02,
         1.16336e+02, 1.16041e+02, 1.15746e+02, 1.15451e+02, 1.15156e+02,
         1.14861e+02, 1.14967e+02, 1.15073e+02, 1.15180e+02, 1.15286e+02,
         1.15392e+02, 1.15498e+02, 1.15604e+02, 1.15711e+02, 1.15817e+02,
         1.15923e+02, 1.15212e+02, 1.14501e+02, 1.13789e+02, 1.13078e+02,
         1.12367e+02, 1.11656e+02, 1.10945e+02, 1.10233e+02, 1.09522e+02,
         1.08811e+02, 1.08865e+02, 1.08920e+02, 1.08974e+02, 1.09028e+02,
         1.09082e+02, 1.09137e+02, 1.09191e+02, 1.09245e+02, 1.09300e+02,
         1.09354e+02, 1.09199e+02, 1.09044e+02, 1.08888e+02, 1.08733e+02,
         1.08578e+02, 1.08423e+02, 1.08268e+02, 1.08112e+02, 1.07957e+02,
         1.07802e+02, 1.07501e+02, 1.07200e+02, 1.06898e+02, 1.06597e+02,
         1.06296e+02, 1.05995e+02, 1.05694e+02, 1.05392e+02, 1.05091e+02,
         1.04790e+02, 1.05080e+02, 1.05370e+02, 1.05660e+02, 1.05950e+02,
         1.06239e+02, 1.06529e+02, 1.06819e+02, 1.07109e+02, 1.07399e+02,
         1.07689e+02, 1.07361e+02, 1.07032e+02, 1.06704e+02, 1.06375e+02,
         1.06047e+02, 1.05719e+02, 1.05390e+02, 1.05062e+02, 1.04733e+02,
         1.04405e+02, 1.04369e+02, 1.04333e+02, 1.04297e+02, 1.04261e+02,
         1.04225e+02, 1.04190e+02, 1.04154e+02, 1.04118e+02, 1.04082e+02,
         1.04046e+02, 1.03641e+02, 1.03237e+02, 1.02832e+02, 1.02428e+02,
         1.02023e+02, 1.01618e+02, 1.01214e+02, 1.00809e+02, 1.00405e+02,
         1.00000e+02, 9.96334e+01, 9.92668e+01, 9.89003e+01, 9.85337e+01,
         9.81671e+01, 9.78005e+01, 9.74339e+01, 9.70674e+01, 9.67008e+01,
         9.63342e+01, 9.62796e+01, 9.62250e+01, 9.61703e+01, 9.61157e+01,
         9.60611e+01, 9.60065e+01, 9.59519e+01, 9.58972e+01, 9.58426e+01,
         9.57880e+01, 9.50778e+01, 9.43675e+01, 9.36573e+01, 9.29470e+01,
         9.22368e+01, 9.15266e+01, 9.08163e+01, 9.01061e+01, 8.93958e+01,
         8.86856e+01, 8.88177e+01, 8.89497e+01, 8.90818e+01, 8.92138e+01,
         8.93459e+01, 8.94780e+01, 8.96100e+01, 8.97421e+01, 8.98741e+01,
         9.00062e+01, 8.99655e+01, 8.99248e+01, 8.98841e+01, 8.98434e+01,
         8.98026e+01, 8.97619e+01, 8.97212e+01, 8.96805e+01, 8.96398e+01,
         8.95991e+01, 8.94091e+01, 8.92190e+01, 8.90290e+01, 8.88389e+01,
         8.86489e+01, 8.84589e+01, 8.82688e+01, 8.80788e+01, 8.78887e+01,
         8.76987e+01, 8.72577e+01, 8.68167e+01, 8.63757e+01, 8.59347e+01,
         8.54936e+01, 8.50526e+01, 8.46116e+01, 8.41706e+01, 8.37296e+01,
         8.32886e+01, 8.33297e+01, 8.33707e+01, 8.34118e+01, 8.34528e+01,
         8.34939e+01, 8.35350e+01, 8.35760e+01, 8.36171e+01, 8.36581e+01,
         8.36992e+01, 8.33320e+01, 8.29647e+01, 8.25975e+01, 8.22302e+01,
         8.18630e+01, 8.14958e+01, 8.11285e+01, 8.07613e+01, 8.03940e+01,
         8.00268e+01, 8.00456e+01, 8.00644e+01, 8.00831e+01, 8.01019e+01,
         8.01207e+01, 8.01395e+01, 8.01583e+01, 8.01770e+01, 8.01958e+01,
         8.02146e+01, 8.04209e+01, 8.06272e+01, 8.08336e+01, 8.10399e+01,
         8.12462e+01, 8.14525e+01, 8.16588e+01, 8.18652e+01, 8.20715e+01,
         8.22778e+01, 8.18784e+01, 8.14791e+01, 8.10797e+01, 8.06804e+01,
         8.02810e+01, 7.98816e+01, 7.94823e+01, 7.90829e+01, 7.86836e+01,
         7.82842e+01, 7.74279e+01, 7.65716e+01, 7.57153e+01, 7.48590e+01,
         7.40027e+01, 7.31465e+01, 7.22902e+01, 7.14339e+01, 7.05776e+01,
         6.97213e+01, 6.99101e+01, 7.00989e+01, 7.02876e+01, 7.04764e+01,
         7.06652e+01, 7.08540e+01, 7.10428e+01, 7.12315e+01, 7.14203e+01,
         7.16091e+01, 7.18831e+01, 7.21571e+01, 7.24311e+01, 7.27051e+01,
         7.29790e+01, 7.32530e+01, 7.35270e+01, 7.38010e+01, 7.40750e+01,
         7.43490e+01, 7.30745e+01, 7.18000e+01, 7.05255e+01, 6.92510e+01,
         6.79765e+01, 6.67020e+01, 6.54275e+01, 6.41530e+01, 6.28785e+01,
         6.16040e+01, 6.24322e+01, 6.32603e+01, 6.40885e+01, 6.49166e+01,
         6.57448e+01, 6.65730e+01, 6.74011e+01, 6.82293e+01, 6.90574e+01,
         6.98856e+01, 7.04057e+01, 7.09259e+01, 7.14460e+01, 7.19662e+01,
         7.24863e+01, 7.30064e+01, 7.35266e+01, 7.40467e+01, 7.45669e+01,
         7.50870e+01, 7.39376e+01, 7.27881e+01, 7.16387e+01, 7.04893e+01,
         6.93398e+01, 6.81904e+01, 6.70410e+01, 6.58916e+01, 6.47421e+01,
         6.35927e+01, 6.18752e+01, 6.01578e+01, 5.84403e+01, 5.67229e+01,
         5.50054e+01, 5.32880e+01, 5.15705e+01, 4.98531e+01, 4.81356e+01,
         4.64182e+01, 4.84569e+01, 5.04956e+01, 5.25344e+01, 5.45731e+01,
         5.66118e+01, 5.86505e+01, 6.06892e+01, 6.27280e+01, 6.47667e+01,
         6.68054e+01, 6.64631e+01, 6.61209e+01, 6.57786e+01, 6.54364e+01,
         6.50941e+01, 6.47518e+01, 6.44096e+01, 6.40673e+01, 6.37251e+01,
         6.33828e+01, 6.34749e+01, 6.35670e+01, 6.36592e+01, 6.37513e+01,
         6.38434e+01, 6.39355e+01, 6.40276e+01, 6.41198e+01, 6.42119e+01,
         6.43040e+01, 6.38188e+01, 6.33336e+01, 6.28484e+01, 6.23632e+01,
         6.18779e+01, 6.13927e+01, 6.09075e+01, 6.04223e+01, 5.99371e+01,
         5.94519e+01, 5.87026e+01, 5.79533e+01, 5.72040e+01, 5.64547e+01,
         5.57054e+01, 5.49562e+01, 5.42069e+01, 5.34576e+01, 5.27083e+01,
         5.19590e+01, 5.25072e+01, 5.30553e+01, 5.36035e+01, 5.41516e+01,
         5.46998e+01, 5.52480e+01, 5.57961e+01, 5.63443e+01, 5.68924e+01,
         5.74406e+01, 5.77278e+01, 5.80150e+01, 5.83022e+01, 5.85894e+01,
         5.88765e+01, 5.91637e+01, 5.94509e+01, 5.97381e+01, 6.00253e+01,
         6.03125e+01])

    # --- post-processing functions ---
    def enhance_sRGB_color_contrast(sRGB_data, color_contrast_multiplier):
        """Enhance the color saturation of of an sRGB image."""
        if contrast_multiplier == 1.0:
            return sRGB_data

        if np.any(sRGB_data < 0.0) or np.any(sRGB_data > 1.0):
            raise ValueError("sRGB data must be in [0, 1] for this routine")

        black_mask = np.logical_and(sRGB_data[:, :, 0] == 0.,
                                    np.logical_and(sRGB_data[:, :, 1] == 0.,
                                                   sRGB_data[:, :, 2] == 0.))

        white_mask = np.logical_and(sRGB_data[:, :, 0] == 1.,
                                    np.logical_and(sRGB_data[:, :, 1] == 1.,
                                                   sRGB_data[:, :, 2] == 1.))

        res = sRGB_data.copy()

        # color contrast enhancement
        # increases sRGB-channel-wise difference between channel and grey value of each pixel
        grey_vals = 0.2989 * sRGB_data[:, :, 0] + 0.5870 * sRGB_data[:, :, 1] + 0.1140 * sRGB_data[:, :, 2]
        grey_vals = grey_vals[:, :, np.newaxis]
        res[:, :, :3] = color_contrast_multiplier * (sRGB_data[:, :, :3] - grey_vals) + grey_vals

        # ensure numerical black and white points are preserved
        res = res.clip(0.0, 1.0)
        res[black_mask, :3] = 0.
        res[white_mask, :3] = 1.

        return res


def _find_closest(A, target):
    # A must be sorted
    idx = np.clip(A.searchsorted(target), 1, len(A)-1)
    idx -= target - A[idx-1] < A[idx] - target
    return idx


def _makeplot(name, block=True, dpi=None):
    import matplotlib.pyplot as plt

    if name is None:
        plt.show(block=block)
        if block:
            plt.close()
        return
    extension = os.path.splitext(name)[1]
    if extension in (".pdf", ".png", ".svg"):
        args = {}
        if dpi is not None:
            args['dpi'] = float(dpi)
        plt.savefig(name, **args)
        plt.close()
    else:
        raise ValueError("file format not understood")


def _limit_xy(**kwargs):
    import matplotlib.pyplot as plt

    x1, x2, y1, y2 = plt.axis()
    x1 = kwargs.pop("xmin", x1)
    x2 = kwargs.pop("xmax", x2)
    y1 = kwargs.pop("ymin", y1)
    y2 = kwargs.pop("ymax", y2)
    plt.axis((x1, x2, y1, y2))


def _register_cmaps():
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    try:
        if _register_cmaps._cmaps_registered:
            return
    except AttributeError:
        _register_cmaps._cmaps_registered = True

    planckcmap = {'red':   ((0., 0., 0.), (.4, 0., 0.), (.5, 1., 1.),
                            (.7, 1., 1.), (.8, .83, .83), (.9, .67, .67),
                            (1., .5, .5)),
                  'green': ((0., 0., 0.), (.2, 0., 0.), (.3, .3, .3),
                            (.4, .7, .7), (.5, 1., 1.), (.6, .7, .7),
                            (.7, .3, .3), (.8, 0., 0.), (1., 0., 0.)),
                  'blue':  ((0., .5, .5), (.1, .67, .67), (.2, .83, .83),
                            (.3, 1., 1.), (.5, 1., 1.), (.6, 0., 0.),
                            (1., 0., 0.))}
    he_cmap = {'red':   ((0., 0., 0.), (.167, 0., 0.), (.333, .5, .5),
                         (.5, 1., 1.), (1., 1., 1.)),
               'green': ((0., 0., 0.), (.5, 0., 0.), (.667, .5, .5),
                         (.833, 1., 1.), (1., 1., 1.)),
               'blue':  ((0., 0., 0.), (.167, 1., 1.), (.333, .5, .5),
                         (.5, 0., 0.), (1., 1., 1.))}
    fd_cmap = {'red':   ((0., .35, .35), (.1, .4, .4), (.2, .25, .25),
                         (.41, .47, .47), (.5, .8, .8), (.56, .96, .96),
                         (.59, 1., 1.), (.74, .8, .8), (.8, .8, .8),
                         (.9, .5, .5), (1., .4, .4)),
               'green': ((0., 0., 0.), (.2, 0., 0.), (.362, .88, .88),
                         (.5, 1., 1.), (.638, .88, .88), (.8, .25, .25),
                         (.9, .3, .3), (1., .2, .2)),
               'blue':  ((0., .35, .35), (.1, .4, .4), (.2, .8, .8),
                         (.26, .8, .8), (.41, 1., 1.), (.44, .96, .96),
                         (.5, .8, .8), (.59, .47, .47), (.8, 0., 0.),
                         (1., 0., 0.))}
    fdu_cmap = {'red':   ((0., 1., 1.), (0.1, .8, .8), (.2, .65, .65),
                          (.41, .6, .6), (.5, .7, .7), (.56, .96, .96),
                          (.59, 1., 1.), (.74, .8, .8), (.8, .8, .8),
                          (.9, .5, .5), (1., .4, .4)),
                'green': ((0., .9, .9), (.362, .95, .95), (.5, 1., 1.),
                          (.638, .88, .88), (.8, .25, .25), (.9, .3, .3),
                          (1., .2, .2)),
                'blue':  ((0., 1., 1.), (.1, .8, .8), (.2, 1., 1.),
                          (.41, 1., 1.), (.44, .96, .96), (.5, .7, .7),
                          (.59, .42, .42), (.8, 0., 0.), (1., 0., 0.))}
    pm_cmap = {'red':   ((0., 1., 1.), (.1, .96, .96), (.2, .84, .84),
                         (.3, .64, .64), (.4, .36, .36), (.5, 0., 0.),
                         (1., 0., 0.)),
               'green': ((0., .5, .5), (.1, .32, .32), (.2, .18, .18),
                         (.3, .8, .8),  (.4, .2, .2), (.5, 0., 0.),
                         (.6, .2, .2), (.7, .8, .8), (.8, .18, .18),
                         (.9, .32, .32), (1., .5, .5)),
               'blue':  ((0., 0., 0.), (.5, 0., 0.), (.6, .36, .36),
                         (.7, .64, .64), (.8, .84, .84), (.9, .96, .96),
                         (1., 1., 1.))}

    mpl.colormaps.register(cmap=LinearSegmentedColormap("Planck-like", planckcmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("High Energy", he_cmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("Faraday Map", fd_cmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("Faraday Uncertainty", fdu_cmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("Plus Minus", pm_cmap))


def _extract_list_kwargs(kwargs, keys, n):
    tmp = {}
    for kk in keys:
        val = kwargs.pop(kk, None)
        tmp[kk] = val if isinstance(val, list) else n*[val]
    with_legend = "label" in keys and any(ll is not None for ll in tmp["label"])
    return [{kk: vv[i] for kk, vv in tmp.items()} for i in range(n)], with_legend


def _plot_history(f, ax, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, date2num

    for i, fld in enumerate(f):
        if not isinstance(fld, EnergyHistory):
            raise TypeError
    label = kwargs.pop("label", None)
    if not isinstance(label, list):
        label = [label] * len(f)
    alpha = kwargs.pop("alpha", None)
    if not isinstance(alpha, list):
        alpha = [alpha] * len(f)
    color = kwargs.pop("color", None)
    if not isinstance(color, list):
        color = [color] * len(f)
    size = kwargs.pop("s", None)
    if not isinstance(size, list):
        size = [size] * len(f)
    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))

    skip_timestamp_conversion = kwargs.pop("skip_timestamp_conversion", False)
    energy_differences = kwargs.pop("plot_energy_differences", False)

    plt.xscale(kwargs.pop("xscale", "linear"))
    default_yscale = 'linear' if not energy_differences else 'log'
    plt.yscale(kwargs.pop("yscale", default_yscale))

    mi, ma = np.inf, -np.inf

    for i, fld in enumerate(f):
        kwargs = {'alpha': alpha[i], 's': size[i], 'color': color[i]}

        if skip_timestamp_conversion:
            xcoord = fld.time_stamps
        else:
            xcoord = date2num([dt.fromtimestamp(ts) for ts in fld.time_stamps])

        if not energy_differences:
            ycoord = fld.energy_values
            ax.scatter(xcoord, ycoord, label=label[i], **kwargs)
        else:
            E = np.array(fld.energy_values)
            dE = E[1:] - E[:-1]
            xcoord = np.array(xcoord[1:])
            idx_pos = (dE > 0)
            idx_neg = (dE < 0)
            label_pos = label[i] + ' (pos)' if label[i] is not None else None
            label_neg = label[i] + ' (neg)' if label[i] is not None else None
            ax.scatter(xcoord[idx_pos], dE[idx_pos], marker='^',
                       label=label_pos, **kwargs)
            ax.scatter(xcoord[idx_neg], dE[idx_neg], marker='v',
                       label=label_neg, **kwargs)

        mi, ma = min([min(xcoord), mi]), max([max(xcoord), ma])

    delta = (ma-mi)*0.05
    if delta == 0.:
        delta = 1.
    ax.set_xlim((mi-delta, ma+delta))
    if not skip_timestamp_conversion:
        xfmt = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(xfmt)
    _limit_xy(**kwargs)
    if label != ([None]*len(f)):
        plt.legend(loc="upper right")


class _UnsupportedInputError(ValueError): pass    # Exception class for lazy plotting function discovery


def _plot1D(f, ax, **kwargs):
    import matplotlib.pyplot as plt

    for i, fld in enumerate(f):
        if not isinstance(fld, Field):
            raise TypeError("incorrect data type")
        if i == 0:
            dom = fld.domain
            if len(dom) != 1:
                raise _UnsupportedInputError("input field must have exactly one domain")
            if len(dom.shape) != 1:
                raise _UnsupportedInputError("input field must have exactly one dimension")
        else:
            check_object_identity(fld.domain, dom)
    dom = dom[0]

    if not isinstance(dom, (RGSpace, PowerSpace)):
        raise _UnsupportedInputError(f"Field type not (yet) supported: {dom}")

    add_kwargs, with_legend = _extract_list_kwargs(kwargs, ("label", "alpha", "color", "linewidth"), len(f))

    if isinstance(dom, RGSpace):
        plt.yscale(kwargs.pop("yscale", "linear"))
        npoints = dom.shape[0]
        dist = dom.distances[0]
        xcoord = np.arange(npoints, dtype=np.float64)*dist
        for i, fld in enumerate(f):
            ycoord = fld.val
            plt.plot(xcoord, ycoord, **add_kwargs[i])
    elif isinstance(dom, PowerSpace):
        plt.xscale(kwargs.pop("xscale", "log"))
        plt.yscale(kwargs.pop("yscale", "log"))
        xcoord = dom.k_lengths
        for i, fld in enumerate(f):
            ycoord = fld.val_rw()
            ycoord[0] = ycoord[1]
            plt.plot(xcoord, ycoord, **add_kwargs[i])
    else:
        raise RuntimeError("This point should never be reached")

    _limit_xy(**kwargs)
    if with_legend:
        ax.legend(loc="upper right")


def plottable2D(fld, f_space=1):
    dom = fld.domain
    if not isinstance(dom, DomainTuple) or len(dom) > 2:
        return False
    if f_space not in [0, 1]:
        return False
    x_space = 0
    if len(dom) == 2:
        x_space = 1 - f_space
        if (not isinstance(dom[f_space], RGSpace)) or len(dom[f_space].shape) != 1:
            return False
    elif len(dom) == 0:
        return False
    if not isinstance(dom[x_space], (RGSpace, HPSpace, GLSpace)):
        return False
    if isinstance(dom[x_space], RGSpace) and not len(dom[x_space].shape) == 2:
        return False
    return True


def _plot2D(f, ax, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from .sugar import makeField

    if len(f) != 1:
        raise _UnsupportedInputError("Can plot only one 2d field")
    f = f[0]
    dom = f.domain

    # check for multifrequency plotting
    have_rgb, rgb = False, None
    if len(dom) == 1:
        x_space = 0
    elif len(dom) == 2:
        f_space = kwargs.pop("freq_space_idx", 1)
        x_space = 1 - f_space

        # Only one frequency?
        n_freqs = dom[f_space].shape[0]
        if n_freqs == 1:
            f = makeField(dom[x_space], f.val.squeeze(axis=dom.axes[f_space]))
        else:
            # Need multifrequency plotting
            val = f.val
            if f_space == 0:
                val = np.moveaxis(val, 0, -1)
            mf_to_rgb = SpectrumToRGBProjector()
            mf_to_rgb.specify_input_spectrum_bins_via_center_energy_and_width(
                # by default assume linearly spaced energy bins
                center_energies = kwargs.pop('f_space_bin_energies', np.arange(n_freqs)),
                widths = kwargs.pop('f_space_bin_widths', np.ones(n_freqs)))
            f_space_saturation_flux = kwargs.pop('f_space_saturation_flux', None)
            if f_space_saturation_flux is not None:
                mf_to_rgb.set_saturation_flux(f_space_saturation_flux)
            rgb = mf_to_rgb.project_total_spectral_bin_flux(val)
            have_rgb = True
    else:  # "DomainTuple can only have one or two entries.
        raise _UnsupportedInputError('Plotting routine cannot handle DomainTuples longer than two :(')

    dom = dom[x_space]
    aspect = kwargs.pop("aspect", None)

    if not have_rgb:
        cmap = kwargs.pop("cmap", plt.rcParams['image.cmap'])
        norm = kwargs.pop("norm", None)

    if isinstance(dom, RGSpace):
        nx, ny = dom.shape
        dx, dy = dom.distances
        if have_rgb:
            im = ax.imshow(
                np.moveaxis(rgb, 0, 1), extent=[0, nx*dx, 0, ny*dy], origin="lower", aspect=aspect)
        else:
            im = ax.imshow(
                f.val.T, extent=[0, nx*dx, 0, ny*dy], origin="lower", aspect=aspect,
                cmap=cmap, vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"), norm=norm)
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        _limit_xy(**kwargs)
        return
    elif isinstance(dom, (HPSpace, GLSpace)):
        xsize = kwargs.pop('xsize', 800)
        projection = kwargs.pop('projection', 'mollweide')
        val = rgb if have_rgb else f.val
        res = project_spherical_data_to_2d(val, dom, projection, xsize, have_rgb)
        plt.axis('off')
        if have_rgb:
            plt.imshow(res, origin="lower")
        else:
            plt.imshow(res, origin="lower", cmap=cmap, norm=norm,
                       vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))
            plt.colorbar(orientation="horizontal")
        return
    raise _UnsupportedInputError("Field type not (yet) supported")


def _plotHist(f, ax, **kwargs):
    add_kwargs, with_legend = _extract_list_kwargs(kwargs, ("label", "alpha", "color", "range"),
                                                   len(f))
    add_kwargs2 = {
        "log": kwargs.pop("log", False),
        "density": kwargs.pop("density", False),
        "bins": kwargs.pop("bins", 50)
    }
    for i, fld in enumerate(f):
        ax.hist(fld.val.ravel(), **add_kwargs[i], **add_kwargs2)
    if with_legend:
        ax.legend(loc="upper right")


def _plot(f, ax, **kwargs):
    _register_cmaps()
    if isinstance(f, (Field, EnergyHistory)):
        f = [f]
    f = list(f)
    if len(f) == 0:
        raise ValueError("need something to plot")
    if isinstance(f[0], EnergyHistory):
        _plot_history(f, ax, **kwargs)
        return
    if not isinstance(f[0], Field):
        raise TypeError("incorrect data type")

    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))
    try:
        _plot1D(f, ax, **kwargs)
        return
    except _UnsupportedInputError:
        pass
    try:
        _plot2D(f, ax, **kwargs)
        return
    except _UnsupportedInputError:
        pass
    _plotHist(f, ax, **kwargs)


class Plot:
    def __init__(self):
        self._plots = []
        self._kwargs = []

    def add(self, f, **kwargs):
        """Add a figure to the current list of plots.

        Notes
        -----
        After doing one or more calls `add()`, one needs to call `output()` to
        show or save the plot.

        Parameters
        ----------
        f : :class:`nifty8.field.Field` or list of :class:`nifty8.field.Field` or None
            If `f` is a single Field, it must be defined on a single `RGSpace`,
            `PowerSpace`, `HPSpace`, `GLSpace`.
            If it is a list, all list members must be Fields defined over the
            same one-dimensional `RGSpace` or `PowerSpace`.
            If `f` is `None`, an empty panel will be displayed.

        Optional Parameters
        -------------------
        title: string
            Title of the plot.
        xlabel: string
            Label for the x axis.
        ylabel: string
            Label for the y axis.
        [xyz]min, [xyz]max: float
            Limits for the values to plot.
        cmap: string
            Color map to use for the plot (if it is a 2D plot).
        linewidth: float or list of floats
            Line width.
        label: string of list of strings
            Annotation string.
        alpha: float or list of floats
            Transparency value.
        freq_space_idx: int
            for multi-frequency plotting: index of frequency space in domain
        """
        if f is None:
            self._plots.append(None)
            self._kwargs.append({})
            return
        if isinstance(f, (MultiField, Field, EnergyHistory)):
            f = [f]
        if hasattr(f, "__len__") and all(isinstance(ff, MultiField) for ff in f):
            for kk in f[0].domain.keys():
                self._plots.append([ff[kk] for ff in f])
                mykwargs = kwargs.copy()
                if 'title' in kwargs:
                    mykwargs['title'] = "{} {}".format(kk, kwargs['title'])
                else:
                    mykwargs['title'] = "{}".format(kk)
                self._kwargs.append(mykwargs)
            return

        if isinstance(f[0], EnergyHistory):
            dom = None
        else:
            dom = f[0].domain

        if isinstance(dom, DomainTuple) \
                and any(isinstance(dd, RGSpace) for dd in dom) \
                and not "freq_space_idx" in kwargs \
                and not plottable2D(f[0], kwargs.get("freq_space_idx", 1)):
            from .sugar import makeField

            dims = [len(dd.shape) for dd in dom]
            # One space is 2d, the rest is 1d
            if np.sum(np.array(dims) == 2) == 1 and np.sum(np.array(dims) == 1) == len(dims) - 1:
                twod_index = dims.index(2)
                sizes = [dd.size for dd in dom]
                del(sizes[twod_index])
                for multi_index in product(*[tuple(range(ii)) for ii in sizes]):
                    multi_index = list(multi_index)
                    multi_index.insert(twod_index, slice(None))
                    multi_index.insert(twod_index, slice(None))
                    for ifield in range(len(f)):
                        arr = f[ifield].val[tuple(multi_index)]
                        myassert(arr.ndim == 2)
                        self._plots.append([makeField(dom[twod_index], arr)])
                        self._kwargs.append(kwargs)
                return
        self._plots.append(f)
        self._kwargs.append(kwargs)

    def output(self, **kwargs):
        """Plot the accumulated list of figures.

        Parameters
        ----------
        title: string
            Title of the full plot.
        nx, ny: int
            Number of subplots to use in x- and y-direction.
            Default: square root of the numer of plots, rounded up.
        xsize, ysize: float
            Dimensions of the full plot in inches. Default: 6.
        name: string
            If left empty, the plot will be shown on the screen,
            otherwise it will be written to a file with the given name.
            Supported extensions: .png and .pdf. Default: None.
        block: bool
            Override the blocking behavior of the non-interactive plotting
            mode. The plot will not be closed in this case but is left open!
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warn("Since matplotlib is not installed, NIFTy will not generate any plots.")
            return

        nplot = len(self._plots)
        fig = plt.figure()
        if "title" in kwargs:
            plt.suptitle(kwargs.pop("title"))
        nx = kwargs.pop("nx", 0)
        ny = kwargs.pop("ny", 0)
        if nx == ny == 0:
            ny = int(np.ceil(np.sqrt(nplot)))
            nx = int(np.ceil(nplot/ny))
            myassert(nx*ny >= nplot)
        elif nx == 0:
            nx = int(np.ceil(nplot/ny))
        elif ny == 0:
            ny = int(np.ceil(nplot/nx))
        if nx*ny < nplot:
            raise ValueError(
                'Figure dimensions not sufficient for number of plots. '
                'Available plot slots: {}, number of plots: {}'
                .format(nx*ny, nplot))
        xsize = kwargs.pop("xsize", 6*nx)
        ysize = kwargs.pop("ysize", 6*ny)
        fig.set_size_inches(xsize, ysize)
        for i in range(nplot):
            if self._plots[i] is None:
                continue
            ax = fig.add_subplot(ny, nx, i+1)
            self._kwargs[i].setdefault('xsise', xsize / nx)
            self._kwargs[i].setdefault('ysize', ysize / ny)
            _plot(self._plots[i], ax, **self._kwargs[i])
        fig.tight_layout()
        _makeplot(kwargs.pop("name", None),
                  block=kwargs.pop("block", True),
                  dpi=kwargs.pop("dpi", None))
