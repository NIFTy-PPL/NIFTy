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
    ysize = xsize//2
    res = np.full(shape=(ysize, xsize), fill_value=np.nan, dtype=np.float64)
    xc, yc = (xsize-1)*0.5, (ysize-1)*0.5
    u, v = np.meshgrid(np.arange(xsize), np.arange(ysize))
    u, v = 2*(u-xc)/(xc/1.02), (v-yc)/(yc/1.02)

    mask = np.where((u*u*0.25 + v*v) <= 1.)
    t1 = v[mask]
    theta = 0.5*np.pi-(
        np.arcsin(2/np.pi*(np.arcsin(t1) + t1*np.sqrt((1.-t1)*(1+t1)))))
    phi = -0.5*np.pi*u[mask]/np.maximum(np.sqrt((1-t1)*(1+t1)), 1e-6)
    phi = np.where(phi < 0, phi+2*np.pi, phi)
    return res, mask, theta, phi


def _hammer_helper(xsize):
    xsize = int(xsize)
    ysize = xsize//2
    res = np.full(shape=(ysize, xsize), fill_value=np.nan, dtype=np.float64)
    xc, yc = (xsize-1)*0.5, (ysize-1)*0.5
    u, v = np.meshgrid(np.arange(xsize), np.arange(ysize))
    u, v = 2*(u-xc)/(xc/1.02), (v-yc)/(yc/1.02)

    u *= np.sqrt(2)
    v *= np.sqrt(2)

    mask = np.where((u*u/8 + v*v/2) <= 1.)
    umask, vmask = u[mask], v[mask]

    zmask = np.sqrt(1-umask*umask/16 - vmask*vmask/4)
    longitude = 2*np.arctan(zmask*umask / 2 / (2*zmask*zmask - 1.))
    latitude = np.arcsin(zmask*vmask)

    theta = np.pi/2 - latitude
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
        res = np.full(shape=res.shape+(3,), fill_value=1., dtype=np.float64)

    if isinstance(domain, HPSpace):
        from ducc0.healpix import Healpix_Base
        ptg = np.empty((phi.size, 2), dtype=np.float64)
        ptg[:, 0] = theta
        ptg[:, 1] = phi
        base = Healpix_Base(int(np.sqrt(domain.size//12)), "RING")
        res[mask] = val[base.ang2pix(ptg)]
    else:
        from ducc0.misc import GL_thetas
        ra = np.linspace(0, 2*np.pi, domain.nlon+1)
        dec = GL_thetas(domain.nlat)
        ilat = _find_closest(dec, theta)
        ilon = _find_closest(ra, phi)
        ilon = np.where(ilon == domain.nlon, 0, ilon)
        res[mask] = val[ilat*domain.nlon + ilon]

    return res


class MultiFrequencyToRGBProjector:
    """Class to facilitate projections of multi-frequency fields to sRGBs color space.

    The fields frequency/energy domain is mapped into the visible light spectral range.
    The thusly mapped images are encoded into the sRGB color space based on a model of
    humans color perception.

    For comparable plots of multiple component reconstructions, a fixed white point
    and black point can be set at initialization time. These can be overridden for
    individual inputs if needed.

    To apply the transformation, use the method :func:`transform`.
    """

    _EQUIVALENT_RGB_INTENSITIES_380nm_TO_780nm = np.array(
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

    _WAVELENGTH_MIN_TABLE = 380.
    _WAVELENGTH_MAX_TABLE = 780.
    _WAVELENGTH_MIN_MAPPABLE = 400.
    _WAVELENGTH_MAX_MAPPABLE = 700.

    _MATRIX_SRGB_D65 = np.array(
            [[3.2404542, -1.5371385, -0.4985314],
             [-0.9692660,  1.8760108,  0.0415560],
             [0.0556434, -0.2040259,  1.0572252]])

    _mapping_f_space_bins_to_rgb = None
    _dynamic_range = None
    _brightness_scale_anchor = None

    def __init__(self, f_space_bin_energies, dynamic_range=1e3, brightness_scale_anchor=None,
                 map_energies_logarithmically=False):
        """The initializtion function sets global preferences for the transformations
        made with the created :class:`MultiFrequencyToRGBProjector` class.

        To set an absolute white point for all plots, set the `brightness_scale_anchor`
        keyword argument. If not set, the white point is chosen for each image individually.

        The black point is set indirectly via the `dynamic_range` parameter.
        It determines the ratio between white point intensity and black point intensity.
        Given the default setting of 1000, values below one 1000th of the white point
        brightness are shown as black.

        To determine a mapping of the multi-frequency values into the visible spectrum
        range, the frequency space bin energies need to be given.
        Users can choose between a linear or logarithmic mapping of input energies to
        visible spectrum photon energies.

        Parameters
        ----------
        f_space_bin_energies : :class:`numpy.ndarray`, list of float
            Reference energies of the frequency/energy domain bins.
        dynamic_range : float, postitive
            Dynamic range to be plotted. Sets the ratio of the white and black point on
            a linear scale. Default: 1000.
        brightness_scale_anchor : float, positive
            Intensity anchor for the white point. If set, all plots share the same
            white and black point. If `None`, white points are chosen for each image
            individually. Default: `None`.
        map_energies_logarithmically : boolean
            Whether to use linear or logarithmic mapping of bin energies to visible
            ligth wavelengths. Default: `False`.
        """
        self._dynamic_range = self._check_pos_scalar(dynamic_range, "dynamic_range")
        self._brightness_scale_anchor = None if brightness_scale_anchor is None else \
            self._check_pos_scalar(brightness_scale_anchor, "brightness_scale_anchor")

        self.set_f_space_bin_to_rgb_mapping(f_space_bin_energies, log_mapping=map_energies_logarithmically)

    def transform(self, spectral_data, override_dynamic_range=False, override_brightness_scale_anchor=False):
        """Projects spherical data `val` onto a two-dimensional plane.

        Parameters
        ----------
        spectral_data : :class:`numpy.ndarray`
            Values of the field to be projected.
            The spectral dimension is expected to be the last dimension of the array.
        override_dynamic_range : float, postitive
            If set, overrides the instance-wide setting of the dynamic range to be plotted.
            Default: `False`.
        override_brightness_scale_anchor : float, positive
            If set, overrides the instance-wide setting of the intensity anchor (white point).
            Default: `False`.

        Returns
        -------
        sRGB_data : :class:`numpy.ndarray`
            The transformed image, containing the sRGB values in the last dimension.
        """
        # input checks
        if (spectral_data < 0.).any():
            raise ValueError("Only positive data supported")

        dynamic_range = self._check_override(override_dynamic_range, "dynamic range", default=self._dynamic_range)
        brightness_scale_anchor = self._check_override(override_brightness_scale_anchor,
                                                       "brightness scale anchor",
                                                       default=self._brightness_scale_anchor)
        n_freqs = spectral_data.shape[-1]
        if self._mapping_f_space_bins_to_rgb.shape[0] != n_freqs:
            raise ValueError("Projector initialized with incompatible f_space.")

        # processing of data
        tgt_shp = spectral_data.shape[:-1]+(3,)
        spectral_data = spectral_data.reshape((-1, n_freqs))

        raw_rgb_data = np.tensordot(spectral_data, self._mapping_f_space_bins_to_rgb, axes=[1, 0])

        max_raw_rgb_data = raw_rgb_data.max() if brightness_scale_anchor is None else brightness_scale_anchor
        min_raw_rgb_data = max_raw_rgb_data / dynamic_range
        raw_rgb_data_log = self._to_logscale(raw_rgb_data, min_raw_rgb_data, max_raw_rgb_data)

        sRGB_data = self.transform_raw_rgb_values_to_sRGB(raw_rgb_data_log)
        sRGB_data = sRGB_data.reshape(tgt_shp)

        # ensuring outputs lie between zero and one
        # LP FIXME: this clipping might lead to data-dependent white point shifts
        # We could replace it with re-normalization, but this reduces color saturation
        sRGB_data = sRGB_data.clip(0., 1.)

        return sRGB_data

    def set_f_space_bin_to_rgb_mapping(self, f_space_bin_energies, log_mapping=True):
        """Sets the frequency/energy domain bin to RGB color mapping to be used for all conversions.

        Parameters
        ----------
        f_space_bin_energies : :class:`numpy.ndarray`, list of float
            Reference energies of the frequency/energy domain bins.
        log_mapping : boolean
            Whether to use linear or logarithmic mapping of bin energies to visible
            ligth wavelengths. Default: `False`.
        """
        E_vis = self.map_f_space_bin_energies_to_visible_range(f_space_bin_energies,
                                                                log_mapping=log_mapping)
        self._mapping_f_space_bins_to_rgb = self.get_equivalent_rgb_intensity(1./E_vis)

    def map_f_space_bin_energies_to_visible_range(self, f_space_bin_energies, log_mapping=False):
        """Maps given bin energy values into the visible spectrum energy range.

        Parameters
        ----------
        f_space_bin_energies : :class:`numpy.ndarray`, list of float
            Reference energies of the frequency/energy domain bins.
        log_mapping : boolean
            Whether to use linear or logarithmic mapping of bin energies to visible
            ligth wavelengths. Default: `False`.

        Returns
        -------
        e_vis : :class:`numpy.ndarray`
            Mapped energies of the given frequency bins.
        """
        E0_vis, E1_vis = 1./self._WAVELENGTH_MAX_MAPPABLE, 1./self._WAVELENGTH_MIN_MAPPABLE

        if log_mapping:
            if (f_space_bin_energies <= 0.).any():
                raise ValueError("log mapping of energies only works with positive energies")
            inp = np.log(f_space_bin_energies)
        else:
            inp = f_space_bin_energies

        inp_min = np.min(inp)
        inp_max = np.max(inp)

        return E0_vis + (inp - inp_min) / (inp_max - inp_min) * (E1_vis - E0_vis)

    def get_equivalent_rgb_intensity(self, wavelength):
        """Linearly interpolate equivalent RGB intensities for given wavelengths.

        Parameters
        ----------
        wavelength : int, float, :class:`numpy.ndarray`
            Wavelength(s) for which to compute the equivalent raw RGB intensities.

        Returns
        -------
        rgb_intensities : :class:`numpy.ndarray`
            Equivalent RGB intensities for the requested wavelength(s).
        """
        # convenience functionality: recursively process numpy arrays
        if isinstance(wavelength, np.ndarray) and len(wavelength.shape) >= 1:
            return np.array([self.get_equivalent_rgb_intensity(wl) for wl in wavelength])

        # assure wavelength is a scalar
        if not self._is_scalar(wavelength):
            raise ValueError("not a scalar: " + str(wavelength))

        # wavelengths outside the table range get mapped to the table ends (dark)
        if wavelength <= self._WAVELENGTH_MIN_TABLE:
            return self._EQUIVALENT_RGB_INTENSITIES_380nm_TO_780nm[:, 0]
        if wavelength >= self._WAVELENGTH_MAX_TABLE:
            return self._EQUIVALENT_RGB_INTENSITIES_380nm_TO_780nm[:, -1]

        delta_wavelength_table = self._WAVELENGTH_MAX_TABLE - self._WAVELENGTH_MIN_TABLE
        rel_wavelength = (wavelength - self._WAVELENGTH_MIN_TABLE) / delta_wavelength_table

        length_table = self._EQUIVALENT_RGB_INTENSITIES_380nm_TO_780nm.shape[1]
        precise_position = rel_wavelength * (length_table - 1)
        idx_table = int(np.floor(precise_position))

        weight = 1. - (precise_position - idx_table)
        res = weight * self._EQUIVALENT_RGB_INTENSITIES_380nm_TO_780nm[:, idx_table]
        res += (1. - weight) * self._EQUIVALENT_RGB_INTENSITIES_380nm_TO_780nm[:, idx_table + 1]
        return res

    def transform_raw_rgb_values_to_sRGB(self, raw_rgb):
        """Transform raw RGB values into the sRGB space.

        Parameters
        ----------
        raw_rgb : :class:`numpy.ndarray`
            Raw RGB values

        Returns
        -------
        res_sRGB : :class:`numpy.ndarray`
            Corresponding sRGB values.
        """
        tmp = np.tensordot(self._MATRIX_SRGB_D65, raw_rgb, axes=(1, 1)).T
        tmp = tmp.clip(0., None)  # remove negative values produce in step above
        return self._sRGB_gammacorr(tmp)

    def _sRGB_gammacorr(self, inp):
        """Perform gamma correction according to sRGB standard."""
        mask = np.zeros(inp.shape, dtype=np.float64)
        mask[inp <= 0.0031308] = 1.
        r1 = 12.92*inp
        a = 0.055
        r2 = (1 + a) * (np.maximum(inp, 0.0031308) ** (1/2.4)) - a
        return r1*mask + r2*(1.-mask)

    def _to_logscale(self, arr, vmin, vmax):
        res = arr.clip(vmin, vmax)
        res = np.log(res/vmin)  # >= 0. by design
        res /= np.log(vmax/vmin)  # <= 1. by design
        return res

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

    mpl.colormaps.register(cmap=LinearSegmentedColormap("Planck-like",
                                                        planckcmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("High Energy",
                                                        he_cmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("Faraday Map",
                                                        fd_cmap))
    mpl.colormaps.register(cmap=LinearSegmentedColormap("Faraday Uncertainty",
                                                        fdu_cmap))
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
            mf_to_rgb = MultiFrequencyToRGBProjector(
                # by default assume linearly spaced energy bins
                f_space_bin_energies = kwargs.pop('f_space_bin_energies', np.arange(n_freqs)),
                dynamic_range = kwargs.pop('dynamic_range', 1e3),
                brightness_scale_anchor = kwargs.pop('brightness_scale_anchor', None),
                map_energies_logarithmically = kwargs.pop('map_energies_logarithmically', False)
            )
            rgb = mf_to_rgb.transform(val)
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
