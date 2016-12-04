# NIFTY (Numerical Information Field Theory) has been developed at the
# Max-Planck-Institute for Astrophysics.
#
# Copyright (C) 2015 Max-Planck-Society
#
# Author: Marco Selig
# Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
    ..                  __   ____   __
    ..                /__/ /   _/ /  /_
    ..      __ ___    __  /  /_  /   _/  __   __
    ..    /   _   | /  / /   _/ /  /   /  / /  /
    ..   /  / /  / /  / /  /   /  /_  /  /_/  /
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  lm
    ..                               /______/

    NIFTY submodule for grids on the two-sphere.

"""
from __future__ import division

import numpy as np

import d2o
from keepers import Versionable

from nifty.spaces.space import Space
from nifty.config import nifty_configuration as gc, \
                         dependency_injector as gdi

hp = gdi.get('healpy')


class HPSpace(Versionable, Space):
    """
        ..        __
        ..      /  /
        ..     /  /___    ______
        ..    /   _   | /   _   |
        ..   /  / /  / /  /_/  /
        ..  /__/ /__/ /   ____/  space class
        ..           /__/

        NIFTY subclass for HEALPix discretizations of the two-sphere [#]_.

        Parameters
        ----------
        nside : int
            Resolution parameter for the HEALPix discretization, resulting in
            ``12*nside**2`` pixels.

        See Also
        --------
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.
        lm_space : A class for spherical harmonic components.

        Notes
        -----
        Only powers of two are allowed for `nside`.

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
            Array containing the number `nside`.
        dtype : numpy.dtype
            Data type of the field values, which is always numpy.float64.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array with one element containing the pixel size.
    """

    # ---Overwritten properties and methods---

    def __init__(self, nside=2, dtype=np.dtype('float')):
        """
            Sets the attributes for a hp_space class instance.

            Parameters
            ----------
            nside : int
                Resolution parameter for the HEALPix discretization, resulting
                in ``12*nside**2`` pixels.

            Returns
            -------
            None

            Raises
            ------
            ImportError
                If the healpy module is not available.
            ValueError
                If input `nside` is invaild.

        """
        # check imports
        if not gc['use_healpy']:
            raise ImportError("healpy not available or not loaded.")

        super(HPSpace, self).__init__(dtype)

        self._nside = self._parse_nside(nside)

    # ---Mandatory properties and methods---

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (np.int(12 * self.nside ** 2),)

    @property
    def dim(self):
        return np.int(12 * self.nside ** 2)

    @property
    def total_volume(self):
        return 4 * np.pi

    def copy(self):
        return self.__class__(nside=self.nside,
                              dtype=self.dtype)

    def weight(self, x, power=1, axes=None, inplace=False):
        weight = ((4*np.pi) / (12 * self.nside**2)) ** power

        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x * weight

        return result_x

    def get_distance_array(self, distribution_strategy):
        """
        Calculates distance from center to all the points on the sphere

        Parameters
        ----------
        distribution_strategy: Result d2o's distribution strategy

        Returns
        -------
        dists: distributed_data_object
        """
        dists = d2o.arange(
            start=0, stop=self.shape[0],
            distribution_strategy=distribution_strategy
        )

        # translate distances to 3D unit vectors on a sphere,
        # extract the first entry (simulates the scalar product with (1,0,0))
        # and apply arccos
        dists = dists.apply_scalar_function(
                    lambda z: np.arccos(hp.pix2vec(self.nside, z)[0]),
                    dtype=np.float)

        return dists

    def get_fft_smoothing_kernel_function(self, sigma):
        if sigma is None:
            sigma = np.sqrt(2) * np.pi

        return lambda x: np.exp((-0.5 * x**2) / sigma**2)

    # ---Added properties and methods---

    @property
    def nside(self):
        return self._nside

    def _parse_nside(self, nside):
        nside = int(nside)
        if nside & (nside - 1) != 0 or nside < 2:
            raise ValueError(
                "nside must be positive and a multiple of 2.")
        return nside

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['nside'] = self.nside
        # metadata
        hdf5_group.attrs['dtype'] = self.dtype.name

        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, loopback_get):
        result = cls(
            nside=hdf5_group['nside'][()],
            dtype=np.dtype(hdf5_group.attrs['dtype'])
            )

        return result
