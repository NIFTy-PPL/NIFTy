from __future__ import division

import itertools
import numpy as np

from d2o import arange, STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.spaces.space import Space
from nifty.config import about, nifty_configuration as gc,\
                         dependency_injector as gdi
import nifty.nifty_utilities as utilities

gl = gdi.get('libsharp_wrapper_gl')

GL_DISTRIBUTION_STRATEGIES = DISTRIBUTION_STRATEGIES['global']


class GLSpace(Space):
    """
        ..                 __
        ..               /  /
        ..     ____ __  /  /
        ..   /   _   / /  /
        ..  /  /_/  / /  /_
        ..  \___   /  \___/  space class
        .. /______/

        NIFTY subclass for Gauss-Legendre pixelizations [#]_ of the two-sphere.

        Parameters
        ----------
        nlat : int
            Number of latitudinal bins, or rings.
        nlon : int, *optional*
            Number of longitudinal bins (default: ``2*nlat - 1``).
        dtype : numpy.dtype, *optional*
            Data type of the field values (default: numpy.float64).

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        lm_space : A class for spherical harmonic components.

        Notes
        -----
        Only real-valued fields on the two-sphere are supported, i.e.
        `dtype` has to be either numpy.float64 or numpy.float32.

        References
        ----------
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing the two numbers `nlat` and `nlon`.
        dtype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array containing the pixel sizes.
    """

    # ---Overwritten properties and methods---

    def __init__(self, nlat=2, nlon=None, dtype=np.dtype('float')):
        """
            Sets the attributes for a gl_space class instance.

            Parameters
            ----------
            nlat : int
                Number of latitudinal bins, or rings.
            nlon : int, *optional*
                Number of longitudinal bins (default: ``2*nlat - 1``).
            dtype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.float64).

            Returns
            -------
            None

            Raises
            ------
            ImportError
                If the libsharp_wrapper_gl module is not available.
            ValueError
                If input `nlat` is invaild.

        """
        # check imports
        if not gc['use_libsharp']:
            raise ImportError(about._errors.cstring(
                "ERROR: libsharp_wrapper_gl not available or not loaded."))

        super(GLSpace, self).__init__(dtype)

        self._nlat = self._parse_nlat(nlat)
        self._nlon = self._parse_nlon(nlon)

    # ---Mandatory properties and methods---

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (np.int((self.nlat * self.nlon)),)

    @property
    def dim(self):
        return np.int((self.nlat * self.nlon))

    @property
    def total_volume(self):
        return 4 * np.pi

    def copy(self):
        return self.__class__(nlat=self.nlat,
                              nlon=self.nlon,
                              dtype=self.dtype)

    def weight(self, x, power=1, axes=None, inplace=False):
        axes = utilities.cast_axis_to_tuple(axes, length=1)

        nlon = self.nlon
        nlat = self.nlat

        weight = np.array(list(itertools.chain.from_iterable(
            itertools.repeat(x ** power, nlon)
            for x in gl.vol(nlat))))

        if axes is not None:
            # reshape the weight array to match the input shape
            new_shape = np.ones(len(x.shape), dtype=np.int)
            for index in range(len(axes)):
                new_shape[index] = len(weight)
            weight = weight.reshape(new_shape)

        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x * weight

        return result_x

    def distance_array(self, distribution_strategy):
        dists = arange(
            start=0, stop=self.shape[0],
            distribution_strategy=distribution_strategy
        )

        dists = dists.apply_scalar_function(
            lambda x: self._distance_array_helper(divmod(int(x), self.nlon)),
            dtype=np.float
        )

        return dists

    def _distance_array_helper(self, qr_tuple):
        numerator = np.sqrt(np.sin(qr_tuple[1])**2 +
                            (np.sin(qr_tuple[0]) * np.cos(qr_tuple[1]))**2)
        denominator = np.cos(qr_tuple[0]) * np.cos(qr_tuple[1])

        return np.arctan(numerator / denominator)

    def get_smoothing_kernel_function(self, sigma):
        if sigma is None:
            sigma = np.sqrt(2) * np.pi / self.nlat

        return lambda x: np.exp((-0.5 * x**2) / sigma**2)

    # ---Added properties and methods---

    @property
    def nlat(self):
        return self._nlat

    @property
    def nlon(self):
        return self._nlon

    def _parse_nlat(self, nlat):
        nlat = int(nlat)
        if nlat < 2:
            raise ValueError(about._errors.cstring(
                "ERROR: nlat must be a positive number."))
        elif nlat % 2 != 0:
            raise ValueError(about._errors.cstring(
                "ERROR: nlat must be a multiple of 2."))
        return nlat

    def _parse_nlon(self, nlon):
        if nlon is None:
            nlon = 2 * self.nlat - 1
        else:
            nlon = int(nlon)
            if nlon != 2 * self.nlat - 1:
                about.warnings.cprint(
                    "WARNING: nlon was set to an unrecommended value: "
                    "nlon <> 2*nlat-1.")
        return nlon
