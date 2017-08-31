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

from ..space import Space


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
        total_volume : np.float
            The total volume of the space.
        shape : tuple of np.ints
            The shape of the space's data array.

    """

    # ---Overwritten properties and methods---

    def __init__(self, shape, distances=None, harmonic=False):
        self._harmonic = bool(harmonic)

        super(RGSpace, self).__init__()

        self._shape = self._parse_shape(shape)
        self._distances = self._parse_distances(distances)

# This code is unused but may be useful to keep around if it is ever needed
# again in the future ...

#    def hermitian_fixed_points(self):
#        dimensions = len(self.shape)
#        mid_index = np.array(self.shape)//2
#        ndlist = [1]*dimensions
#        for k in range(dimensions):
#            if self.shape[k] % 2 == 0:
#                ndlist[k] = 2
#        ndlist = tuple(ndlist)
#        fixed_points = []
#        for index in np.ndindex(ndlist):
#            for k in range(dimensions):
#                if self.shape[k] % 2 != 0 and self.zerocenter[k]:
#                    index = list(index)
#                    index[k] = 1
#                    index = tuple(index)
#            fixed_points += [tuple(index * mid_index)]
#        return fixed_points

    def hermitianize_inverter(self, x, axes):
        return x

    # ---Mandatory properties and methods---

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
        return int(reduce(lambda x, y: x*y, self.shape))

    @property
    def total_volume(self):
        return self.dim * reduce(lambda x, y: x*y, self.distances)

    def copy(self):
        return self.__class__(shape=self.shape,
                              distances=self.distances,
                              harmonic=self.harmonic)

    def weight(self, x, power=1, axes=None, inplace=False):
        weight = reduce(lambda x, y: x*y, self.distances) ** np.float(power)
        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x*weight
        return result_x

    def get_distance_array(self):
        """ Calculates an n-dimensional array with its entries being the
        lengths of the vectors from the zero point of the grid.

        Returns
        -------
        numpy.ndarray
            An array containing the distances.

        """

        if (not self.harmonic):
            raise NotImplementedError
        shape = self.shape

        slice_of_first_dimension = slice(0, shape[0])
        dists = self._distance_array_helper(slice_of_first_dimension)

        return dists

    def _distance_array_helper(self, slice_of_first_dimension):
        dk = self.distances
        shape = self.shape

        inds = []
        for a in shape:
            inds += [slice(0, a)]

        cords = np.ogrid[inds]

        dists = (cords[0] - shape[0]//2)*dk[0]
        dists *= dists
        dists = np.fft.ifftshift(dists)
        # only save the individual slice
        dists = dists[slice_of_first_dimension]
        for ii in range(1, len(shape)):
            temp = (cords[ii] - shape[ii] // 2) * dk[ii]
            temp *= temp
            temp = np.fft.ifftshift(temp)
            dists = dists + temp
        dists = np.sqrt(dists)
        return dists

    def get_unique_distances(self):
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
            tmp = self.get_distance_array('not').unique()  # expensive!
            tol = 1e-12*tmp[-1]
            # remove all points that are closer than tol to their right
            # neighbors.
            # I'm appending the last value*2 to the array to treat the
            # rightmost point correctly.
            return tmp[np.diff(np.r_[tmp, 2*tmp[-1]]) > tol]

    def get_natural_binbounds(self):
        if (not self.harmonic):
            raise NotImplementedError
        tmp = self.get_unique_distances()
        return 0.5*(tmp[:-1]+tmp[1:])

    def get_fft_smoothing_kernel_function(self, sigma):
        if (not self.harmonic):
            raise NotImplementedError
        return lambda x: np.exp(-2. * np.pi*np.pi * x*x * sigma*sigma)

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
