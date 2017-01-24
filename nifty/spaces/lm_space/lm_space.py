from __future__ import division

import numpy as np

from nifty.spaces.space import Space

from nifty.config import nifty_configuration as gc,\
                         dependency_injector as gdi

from lm_helper import _distance_array_helper

from d2o import arange

gl = gdi.get('libsharp_wrapper_gl')
hp = gdi.get('healpy')


class LMSpace(Space):
    """
        ..       __
        ..     /  /
        ..    /  /    __ ____ ___
        ..   /  /   /   _    _   |
        ..  /  /_  /  / /  / /  /
        ..  \___/ /__/ /__/ /__/  space class

        NIFTY subclass for spherical harmonics components, for representations
        of fields on the two-sphere.

        Parameters
        ----------
        lmax : int
            Maximum :math:`\ell`-value up to which the spherical harmonics
            coefficients are to be used.
        mmax : int, *optional*
            Maximum :math:`m`-value up to which the spherical harmonics
            coefficients are to be used (default: `lmax`).
        dtype : numpy.dtype, *optional*
            Data type of the field values (default: numpy.complex128).

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.

        Notes
        -----
        Hermitian symmetry, i.e. :math:`a_{\ell -m} = \overline{a}_{\ell m}` is
        always assumed for the spherical harmonics components, i.e. only fields
        on the two-sphere with real-valued representations in position space
        can be handled.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing the two numbers `lmax` and
            `mmax`.
        dtype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Parameter captioning the fact that an :py:class:`lm_space` is
            always discrete.
        vol : numpy.ndarray
            Pixel volume of the :py:class:`lm_space`, which is always 1.
    """

    def __init__(self, lmax, dtype=None):
        """
            Sets the attributes for an lm_space class instance.

            Parameters
            ----------
            lmax : int
                Maximum :math:`\ell`-value up to which the spherical harmonics
                coefficients are to be used.
            mmax : int, *optional*
                Maximum :math:`m`-value up to which the spherical harmonics
                coefficients are to be used (default: `lmax`).
            dtype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.complex128).

            Returns
            -------
            None.

            Raises
            ------
            ImportError
                If neither the libsharp_wrapper_gl nor the healpy module are
                available.
            ValueError
                If input `nside` is invaild.

        """

        # check imports
        if not gc['use_libsharp'] and not gc['use_healpy']:
            raise ImportError(
                "neither libsharp_wrapper_gl nor healpy activated.")

        super(LMSpace, self).__init__(dtype)
        self._lmax = self._parse_lmax(lmax)

    def hermitian_decomposition(self, x, axes=None):
        hermitian_part = x.copy_empty()
        anti_hermitian_part = x.copy_empty()
        hermitian_part[:] = x.real
        anti_hermitian_part[:] = x.imag
        return (hermitian_part, anti_hermitian_part)

        # ---Mandatory properties and methods---

    @property
    def harmonic(self):
        return True

    @property
    def shape(self):
        return (self.dim, )

    @property
    def dim(self):
        l = self.lmax
        # the LMSpace consist of the full triangle (including -m's!),
        # minus two little triangles if mmax < lmax
        # dim = (((2*(l+1)-1)+1)**2/4 - 2 * (l-m)(l-m+1)/2
        # dim = np.int((l+1)**2 - (l-m)*(l-m+1.))
        # We fix l == m
        return np.int((l+1)**2)

    @property
    def total_volume(self):
        # the individual pixels have a fixed volume of 1.
        return np.float(self.dim)

    def copy(self):
        return self.__class__(lmax=self.lmax,
                              mmax=self.mmax,
                              dtype=self.dtype)

    def weight(self, x, power=1, axes=None, inplace=False):
        if inplace:
            return x
        else:
            return x.copy()

    def get_distance_array(self, distribution_strategy):
        dists = arange(start=0, stop=self.shape[0],
                       distribution_strategy=distribution_strategy)

        dists = dists.apply_scalar_function(
            lambda x: _distance_array_helper(x, self.lmax),
            dtype=np.float)

        return dists

    def get_fft_smoothing_kernel_function(self, sigma):
        if sigma is None:
            sigma = np.sqrt(2) * np.pi / (self.lmax + 1)

        return lambda x: np.exp(-0.5 * x * (x + 1) * sigma**2)

    # ---Added properties and methods---

    @property
    def lmax(self):
        return self._lmax

    @property
    def mmax(self):
        return self._lmax

    def _parse_lmax(self, lmax):
        lmax = np.int(lmax)
        if lmax < 1:
            raise ValueError("Negative lmax is not allowed.")
        # exception lmax == 2 (nside == 1)
        if (lmax % 2 == 0) and (lmax > 2):
            self.logger.warn("Unrecommended parameter (lmax <> 2*n+1).")
        return lmax

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['lmax'] = self.lmax
        hdf5_group['dtype'] = self.dtype.name
        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls(
            lmax=hdf5_group['lmax'][()],
            dtype=np.dtype(hdf5_group['dtype'][()])
            )
        return result


    def plot(self):
        pass