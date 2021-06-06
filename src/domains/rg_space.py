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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce

import numpy as np

from ..field import Field
from .structured_domain import StructuredDomain


class RGSpace(StructuredDomain):
    """Represents a regular Cartesian grid.

    Parameters
    ----------
    shape : int or tuple of int
        Number of grid points or numbers of gridpoints along each axis.
    distances : None or float or tuple of float, optional
        Distance between two grid points along each axis.

        By default (distances=None):
          - If harmonic==True, all distances will be set to 1
          - If harmonic==False, the distance along each axis will be
            set to the inverse of the number of points along that axis.

    harmonic : bool, optional
        Whether the space represents a grid in position or harmonic space.
        Default: False.

    Notes
    -----
    Topologically, a n-dimensional RGSpace is a n-Torus, i.e. it has periodic
    boundary conditions.
    """
    _needed_for_hash = ["_rdistances", "_shape", "_harmonic"]

    def __init__(self, shape, distances=None, harmonic=False, _realdistances=None):
        self._harmonic = bool(harmonic)
        if np.isscalar(shape):
            shape = (shape,)
        self._shape = tuple(int(i) for i in shape)
        if min(self._shape) < 0:
            raise ValueError('Negative number of pixels encountered')

        if _realdistances is not None:
            self._rdistances = _realdistances
        else:
            if distances is None:
                self._rdistances = tuple(1. / (np.array(self._shape)))
            elif np.isscalar(distances):
                if self.harmonic:
                    self._rdistances = tuple(
                        1. / (np.array(self._shape) * float(distances)))
                else:
                    self._rdistances = (float(distances),) * len(self._shape)
            else:
                temp = np.empty(len(self.shape), dtype=np.float64)
                temp[:] = distances
                if self._harmonic:
                    temp = 1. / (np.array(self._shape) * temp)
                self._rdistances = tuple(temp)
        self._hdistances = tuple(
            1. / (np.array(self.shape)*np.array(self._rdistances)))
        if min(self._rdistances) <= 0:
            raise ValueError('Non-positive distances encountered')

        self._dvol = float(reduce(lambda x, y: x*y, self.distances))
        self._size = int(reduce(lambda x, y: x*y, self._shape))

    def __repr__(self):
        return ("RGSpace(shape={}, distances={}, harmonic={})"
                .format(self.shape, self.distances, self.harmonic))

    @property
    def harmonic(self):
        return self._harmonic

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def scalar_dvol(self):
        return self._dvol

    def _get_dist_array(self):
        res = np.arange(self.shape[0], dtype=np.float64)
        res = np.minimum(res, self.shape[0]-res)*self.distances[0]
        if len(self.shape) == 1:
            return Field.from_raw(self, res)
        res *= res
        for i in range(1, len(self.shape)):
            tmp = np.arange(self.shape[i], dtype=np.float64)
            tmp = np.minimum(tmp, self.shape[i]-tmp)*self.distances[i]
            tmp *= tmp
            res = np.add.outer(res, tmp)
        return Field.from_raw(self, np.sqrt(res))

    def get_k_length_array(self):
        if (not self.harmonic):
            raise NotImplementedError
        return self._get_dist_array()

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
            tmp = np.zeros(tmp+1, dtype=bool)
            t2 = np.arange(maxdist[0]+1, dtype=np.int64)
            t2 *= t2
            for i in range(1, dimensions):
                t3 = np.arange(maxdist[i]+1, dtype=np.int64)
                t3 *= t3
                t2 = np.add.outer(t2, t3)
            tmp[t2] = True
            return np.sqrt(np.nonzero(tmp)[0])*self.distances[0]
        else:  # do it the hard way
            tmp = self.get_k_length_array().val
            tmp = np.unique(tmp)
            tol = 1e-12*tmp[-1]
            # remove all points that are closer than tol to their right
            # neighbors.
            # I'm appending the last value*2 to the array to treat the
            # rightmost point correctly.
            return tmp[np.diff(np.r_[tmp, 2*tmp[-1]]) > tol]

    @staticmethod
    def _kernel(x, sigma):
        return (x*x * (-2.*np.pi*np.pi*sigma*sigma)).ptw("exp")

    def get_fft_smoothing_kernel_function(self, sigma):
        if (not self.harmonic):
            raise NotImplementedError
        return lambda x: self._kernel(x, sigma)

    def get_conv_kernel_from_func(self, func):
        """Creates a convolution kernel defined by a function.

        Assumes the function to be radially symmetric, e.g. only dependant on
        distance.

        Parameters
        ----------
        func: function
            This function needs to take exactly one argument, which is
            distance from center (in the same units as the RGSpace distances),
            and return the kernel amplitude at that distance.
        """
        from ..operators.harmonic_operators import HarmonicTransformOperator
        if (not self.harmonic):
            raise NotImplementedError
        op = HarmonicTransformOperator(self, self.get_default_codomain())
        dist = op.target[0]._get_dist_array()
        kernel = Field(op.target, func(dist.val))
        kernel = kernel / kernel.s_integrate()
        return op.adjoint_times(kernel.weight(1))

    def get_default_codomain(self):
        """Returns a :class:`RGSpace` object representing the (position or
        harmonic) partner domain of `self`, depending on `self.harmonic`.

        Returns
        -------
        RGSpace
            The partner domain
        """
        return RGSpace(self.shape, None, not self.harmonic, self._rdistances)

    def check_codomain(self, codomain):
        """Raises `TypeError` if `codomain` is not a matching partner domain
        for `self`.
        """
        if not isinstance(codomain, RGSpace):
            raise TypeError("domain is not a RGSpace")

        if self.shape != codomain.shape:
            raise AttributeError("The shapes of domain and codomain must be "
                                 "identical.")

        if self.harmonic == codomain.harmonic:
            raise AttributeError("domain.harmonic and codomain.harmonic must "
                                 "not be the same.")

        # Check if the distances match, i.e. dist' = 1 / (num * dist)
        if not np.all(abs(np.array(self.shape) *
                          np.array(self.distances) *
                          np.array(codomain.distances)-1) < 1e-7):
            raise AttributeError("The grid-distances of domain and codomain "
                                 "do not match.")

    @property
    def distances(self):
        """tuple of float : Distance between grid points along each axis.
        The n-th entry of the tuple is the distance between neighboring
        grid points along the n-th dimension.
        """
        return self._hdistances if self._harmonic else self._rdistances
