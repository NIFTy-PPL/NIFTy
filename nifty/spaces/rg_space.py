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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

"""
    ..                  __   ____   __
    ..                /__/ /   _/ /  /_
    ..      __ ___    __  /  /_  /   _/  __   __
    ..    /   _   | /  / /   _/ /  /   /  / /  /
    ..   /  / /  / /  / /  /   /  /_  /  /_/  /
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  rg
    ..                               /______/

    NIFTY submodule for regular Cartesian grids.

"""
from __future__ import division
from builtins import range
from functools import reduce
import numpy as np
from .space import Space


class RGSpace(Space):
    """
        ..      _____   _______
        ..    /   __/ /   _   /
        ..   /  /    /  /_/  /
        ..  /__/     \____  /  space class
        ..          /______/

        NIFTY subclass for spaces of regular Cartesian grids.

        Parameters
        ----------
        shape : {int, numpy.ndarray}
            Number of grid points or numbers of gridpoints along each axis.
        distances : {float, numpy.ndarray}, *optional*
            Distance between two grid points along each axis
            (default: None).
            If distances==None:
                if harmonic==True, all distances will be set to 1
                if harmonic==False, the distance along each axis will be
                  set to the inverse of the number of points along that
                  axis.
        harmonic : bool, *optional*
        Whether the space represents a grid in position or harmonic space.
            (default: False).

        Attributes
        ----------
        harmonic : bool
            Whether or not the grid represents a position or harmonic space.
        distances : tuple of floats
            Distance between two grid points along the correponding axis.
        dim : np.int
            Total number of dimensionality, i.e. the number of pixels.
        harmonic : bool
            Specifies whether the space is a signal or harmonic space.
        shape : tuple of np.ints
            The shape of the space's data array.

    """

    # ---Overwritten properties and methods---

    def __init__(self, shape, distances=None, harmonic=False):
        super(RGSpace, self).__init__()
        self._needed_for_hash += ["_distances", "_shape", "_harmonic"]

        self._harmonic = bool(harmonic)
        self._shape = self._parse_shape(shape)
        self._distances = self._parse_distances(distances)
        self._dvol = float(reduce(lambda x, y: x*y, self._distances))
        self._dim = int(reduce(lambda x, y: x*y, self._shape))

    def __repr__(self):
        return ("RGSpace(shape=%r, distances=%r, harmonic=%r)"
                % (self.shape, self.distances, self.harmonic))

    @property
    def harmonic(self):
        return self._harmonic

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return self._dim

    def scalar_dvol(self):
        return self._dvol

    def get_k_length_array(self):
        if (not self.harmonic):
            raise NotImplementedError
        res = np.arange(self.shape[0], dtype=np.float64)
        res = np.minimum(res, self.shape[0]-res)*self.distances[0]
        if len(self.shape) == 1:
            return res
        res *= res
        for i in range(1, len(self.shape)):
            tmp = np.arange(self.shape[i], dtype=np.float64)
            tmp = np.minimum(tmp, self.shape[i]-tmp)*self.distances[i]
            tmp *= tmp
            res = np.add.outer(res, tmp)
        return np.sqrt(res)

    def get_unique_k_lengths(self):
        if (not self.harmonic):
            raise NotImplementedError
        dimensions = len(self.shape)
        if dimensions == 1:  # extra easy
            maxdist = self.shape[0]//2
            return np.arange(maxdist+1, dtype=np.float64) * self.distances[0]
        if np.all(self.distances == self.distances[0]):  # shortcut
            maxdist = np.asarray(self.shape)//2
            tmp = np.sum(maxdist*maxdist)
            tmp = np.zeros(tmp+1, dtype=np.bool)
            t2 = np.arange(maxdist[0]+1, dtype=np.int64)
            t2 *= t2
            for i in range(1, dimensions):
                t3 = np.arange(maxdist[i]+1, dtype=np.int64)
                t3 *= t3
                t2 = np.add.outer(t2, t3)
            tmp[t2] = True
            return np.sqrt(np.nonzero(tmp)[0])*self.distances[0]
        else:  # do it the hard way
            tmp = self.get_k_length_array().unique()  # expensive!
            tol = 1e-12*tmp[-1]
            # remove all points that are closer than tol to their right
            # neighbors.
            # I'm appending the last value*2 to the array to treat the
            # rightmost point correctly.
            return tmp[np.diff(np.r_[tmp, 2*tmp[-1]]) > tol]

    @staticmethod
    def _kernel(x, sigma):
        tmp = x*x
        tmp *= -2.*np.pi*np.pi*sigma*sigma
        np.exp(tmp, out=tmp)
        return tmp

    def get_fft_smoothing_kernel_function(self, sigma):
        if (not self.harmonic):
            raise NotImplementedError
        return lambda x: self._kernel(x, sigma)

    def get_default_codomain(self):
        distances = 1. / (np.array(self.shape)*np.array(self.distances))
        return RGSpace(self.shape, distances, not self.harmonic)

    def check_codomain(self, codomain):
        if not isinstance(codomain, RGSpace):
            raise TypeError("domain is not a RGSpace")

        if self.shape != codomain.shape:
            raise AttributeError("The shapes of domain and codomain must be "
                                 "identical.")

        if self.harmonic == codomain.harmonic:
            raise AttributeError("domain.harmonic and codomain.harmonic must "
                                 "not be the same.")

        # Check if the distances match, i.e. dist' = 1 / (num * dist)
        if not np.all(
            np.absolute(np.array(self.shape) *
                        np.array(self.distances) *
                        np.array(codomain.distances) - 1) < 1e-7):
            raise AttributeError("The grid-distances of domain and codomain "
                                 "do not match.")

    # ---Added properties and methods---

    @property
    def distances(self):
        """Distance between two grid points along each axis. It is a tuple
        of positive floating point numbers with the n-th entry giving the
        distances of grid points along the n-th dimension.
        """

        return self._distances

    def _parse_shape(self, shape):
        if np.isscalar(shape):
            shape = (shape,)
        temp = np.empty(len(shape), dtype=np.int)
        temp[:] = shape
        return tuple(temp)

    def _parse_distances(self, distances):
        if distances is None:
            if self.harmonic:
                temp = np.ones_like(self.shape, dtype=np.float64)
            else:
                temp = 1 / np.array(self.shape, dtype=np.float64)
        else:
            temp = np.empty(len(self.shape), dtype=np.float64)
            temp[:] = distances
        return tuple(temp)
