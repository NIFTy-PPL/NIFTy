# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

from __future__ import division

import numpy as np

from nifty.spaces.space import Space

from d2o import arange


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


        Notes:
        ------
        This implementation implicitly sets the mmax parameter to lmax.

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
    """

    def __init__(self, lmax):
        """
            Sets the attributes for an lm_space class instance.

            Parameters
            ----------
            lmax : int
                Maximum :math:`\ell`-value up to which the spherical harmonics
                coefficients are to be used.

            Returns
            -------
            None.

        """

        super(LMSpace, self).__init__()
        self._lmax = self._parse_lmax(lmax)

    def hermitian_decomposition(self, x, axes=None,
                                preserve_gaussian_variance=False):
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
        # the LMSpace consists of the full triangle (including -m's!),
        # minus two little triangles if mmax < lmax
        # dim = (((2*(l+1)-1)+1)**2/4 - 2 * (l-m)(l-m+1)/2
        # dim = np.int((l+1)**2 - (l-m)*(l-m+1.))
        # We fix l == m
        return np.int((l+1)**2)

    @property
    def total_volume(self):
        # the individual pixels have a fixed volume of 1.
        return np.float64(self.dim)

    def copy(self):
        return self.__class__(lmax=self.lmax)

    def weight(self, x, power=1, axes=None, inplace=False):
        if inplace:
            return x
        else:
            return x.copy()

    def get_distance_array(self, distribution_strategy):
        dists = arange(start=0, stop=self.shape[0],
                       distribution_strategy=distribution_strategy)

        dists = dists.apply_scalar_function(
            lambda x: self._distance_array_helper(x, self.lmax),
            dtype=np.float64)

        return dists

    @staticmethod
    def _distance_array_helper(index_array, lmax):
        u = 2*lmax + 1
        index_half = (index_array+np.minimum(lmax, index_array)+1)//2
        m = (np.ceil((u - np.sqrt(u*u - 8*(index_half - lmax)))/2)).astype(int)
        res = (index_half - m*(u - m)//2).astype(np.float64)
        return res

    def get_fft_smoothing_kernel_function(self, sigma):
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
        if lmax < 0:
            raise ValueError("lmax must be >=0.")
        return lmax

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['lmax'] = self.lmax
        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls(
            lmax=hdf5_group['lmax'][()],
            )
        return result
