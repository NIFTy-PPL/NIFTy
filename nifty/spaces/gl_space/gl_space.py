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

import itertools
import numpy as np

from nifty.spaces.space import Space
from nifty.config import dependency_injector as gdi

pyHealpix = gdi.get('pyHealpix')


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
            
        Attributes
        ----------
        dim : np.int
            Total number of dimensionality, i.e. the number of pixels.
        harmonic : bool
            Specifies whether the space is a signal or harmonic space.
        nlat : int
            Number of latitudinal bins (or rings) that are used for this
            pixelization.
        nlon : int, *optional*
            Number of longditudinal bins that are used for this pixelization.
        total_volume : np.float
            The total volume of the space.
        shape : tuple of np.ints
            The shape of the space's data array.

        Raises
        ------
        ValueError
            If input `nlat` or `nlon` is invalid.
        ImportError
            If the pyHealpix module is not available

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        lm_space : A class for spherical harmonic components.

        References
        ----------
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.

    """

    # ---Overwritten properties and methods---

    def __init__(self, nlat, nlon=None):
        if 'pyHealpix' not in gdi:
            raise ImportError(
                "The module pyHealpix is needed but not available.")

        super(GLSpace, self).__init__()

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
                              nlon=self.nlon)

    def weight(self, x, power=1, axes=None, inplace=False):
        nlon = self.nlon
        nlat = self.nlat
        vol = pyHealpix.GL_weights(nlat, nlon) ** power
        weight = np.array(list(itertools.chain.from_iterable(
                          itertools.repeat(x, nlon) for x in vol)))

        if axes is not None:
            # reshape the weight array to match the input shape
            new_shape = np.ones(len(x.shape), dtype=np.int)
            # we know len(axes) is always 1
            new_shape[axes[0]] = len(weight)
            weight = weight.reshape(new_shape)

        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x * weight

        return result_x

    def get_distance_array(self, distribution_strategy):
        raise NotImplementedError

    def get_fft_smoothing_kernel_function(self, sigma):
        raise NotImplementedError

    # ---Added properties and methods---

    @property
    def nlat(self):
        """ Number of latitudinal bins (or rings) that are used for this
        pixelization.
        """

        return self._nlat

    @property
    def nlon(self):
        """ Number of longditudinal bins that are used for this pixelization.
        """

        return self._nlon

    def _parse_nlat(self, nlat):
        nlat = int(nlat)
        if nlat < 1:
            raise ValueError(
                "nlat must be a positive number.")
        return nlat

    def _parse_nlon(self, nlon):
        if nlon is None:
            nlon = 2 * self.nlat - 1
        else:
            nlon = int(nlon)
            if nlon < 1:
                raise ValueError("nlon must be a positive number.")
        return nlon

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['nlat'] = self.nlat
        hdf5_group['nlon'] = self.nlon

        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls(
            nlat=hdf5_group['nlat'][()],
            nlon=hdf5_group['nlon'][()],
            )

        return result
