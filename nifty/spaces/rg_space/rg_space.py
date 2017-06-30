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

import numpy as np

from d2o import distributed_data_object,\
                STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.spaces.space import Space
from functools import reduce


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
        zerocenter : {bool, numpy.ndarray} *optional*
            Whether x==0 (or k==0, respectively) is located in the center of
            the grid (or the center of each axis speparately) or not.
            (default: False).
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
        zerocenter : tuple of bool
            Whether x==0 (or k==0, respectively) is located in the center of
            the grid (or the center of each axis speparately) or not.
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

    def __init__(self, shape, zerocenter=False, distances=None,
                 harmonic=False):
        self._harmonic = bool(harmonic)

        super(RGSpace, self).__init__()

        self._shape = self._parse_shape(shape)
        self._distances = self._parse_distances(distances)
        self._zerocenter = self._parse_zerocenter(zerocenter)

    def hermitian_decomposition(self, x, axes=None,
                                preserve_gaussian_variance=False):
        # compute the hermitian part
        flipped_x = self._hermitianize_inverter(x, axes=axes)
        flipped_x = flipped_x.conjugate()
        # average x and flipped_x.
        hermitian_part = x + flipped_x
        hermitian_part /= 2.

        # use subtraction since it is faster than flipping another time
        anti_hermitian_part = (x-hermitian_part)

        if preserve_gaussian_variance:
            hermitian_part, anti_hermitian_part = \
                self._hermitianize_correct_variance(hermitian_part,
                                                    anti_hermitian_part,
                                                    axes=axes)

        return (hermitian_part, anti_hermitian_part)

    def _hermitianize_correct_variance(self, hermitian_part,
                                       anti_hermitian_part, axes):
        # Correct the variance by multiplying sqrt(2)
        hermitian_part = hermitian_part * np.sqrt(2)
        anti_hermitian_part = anti_hermitian_part * np.sqrt(2)

        # If the dtype of the input is complex, the fixed points lose the power
        # of their imaginary-part (or real-part, respectively). Therefore
        # the factor of sqrt(2) also applies there
        if not issubclass(hermitian_part.dtype.type, np.complexfloating):
            # The fixed points of the point inversion must not be averaged.
            # Hence one must divide out the sqrt(2) again
            # -> Get the middle index of the array
            mid_index = np.array(hermitian_part.shape, dtype=np.int) // 2
            dimensions = mid_index.size
            # Use ndindex to iterate over all combinations of zeros and the
            # mid_index in order to correct all fixed points.
            if axes is None:
                axes = range(dimensions)

            ndlist = [2 if i in axes else 1 for i in range(dimensions)]
            ndlist = tuple(ndlist)
            for i in np.ndindex(ndlist):
                temp_index = tuple(i * mid_index)
                hermitian_part[temp_index] /= np.sqrt(2)
                anti_hermitian_part[temp_index] /= np.sqrt(2)
        return hermitian_part, anti_hermitian_part

    def _hermitianize_inverter(self, x, axes):
        shape = x.shape
        # calculate the number of dimensions the input array has
        dimensions = len(shape)
        # prepare the slicing object which will be used for mirroring
        slice_primitive = [slice(None), ] * dimensions
        # copy the input data
        y = x.copy()

        if axes is None:
            axes = range(dimensions)

        # flip in the desired directions
        for i in axes:
            slice_picker = slice_primitive[:]
            if shape[i] % 2 == 0:
                slice_picker[i] = slice(1, None, None)
            else:
                slice_picker[i] = slice(None)
            slice_picker = tuple(slice_picker)

            slice_inverter = slice_primitive[:]
            if shape[i] % 2 == 0:
                slice_inverter[i] = slice(None, 0, -1)
            else:
                slice_inverter[i] = slice(None, None, -1)
            slice_inverter = tuple(slice_inverter)

            try:
                y.set_data(to_key=slice_picker, data=y,
                           from_key=slice_inverter)
            except(AttributeError):
                y[slice_picker] = y[slice_inverter]
        return y

    # ---Mandatory properties and methods---

    def __repr__(self):
        return ("RGSpace(shape=%r, zerocenter=%r, distances=%r, harmonic=%r)"
                % (self.shape, self.zerocenter, self.distances, self.harmonic))

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
                              zerocenter=self.zerocenter,
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

    def get_distance_array(self, distribution_strategy):
        """ Calculates an n-dimensional array with its entries being the
        lengths of the vectors from the zero point of the grid.

        Parameters
        ----------
        distribution_strategy : str
            The distribution_strategy which shall be used the returned
            distributed_data_object.

        Returns
        -------
        distributed_data_object
            A d2o containing the distances.

        """

        shape = self.shape
        # prepare the distributed_data_object
        nkdict = distributed_data_object(
                        global_shape=shape, dtype=np.float64,
                        distribution_strategy=distribution_strategy)

        if distribution_strategy in DISTRIBUTION_STRATEGIES['slicing']:
            # get the node's individual slice of the first dimension
            slice_of_first_dimension = slice(
                                    *nkdict.distributor.local_slice[0:2])
        elif distribution_strategy in DISTRIBUTION_STRATEGIES['not']:
            slice_of_first_dimension = slice(0, shape[0])
        else:
            raise ValueError(
                "Unsupported distribution strategy")
        dists = self._distance_array_helper(slice_of_first_dimension)
        nkdict.set_local_data(dists)

        return nkdict

    def _distance_array_helper(self, slice_of_first_dimension):
        dk = self.distances
        shape = self.shape

        inds = []
        for a in shape:
            inds += [slice(0, a)]

        cords = np.ogrid[inds]

        dists = (cords[0] - shape[0]//2)*dk[0]
        dists *= dists
        # apply zerocenterQ shift
        if not self.zerocenter[0]:
            dists = np.fft.ifftshift(dists)
        # only save the individual slice
        dists = dists[slice_of_first_dimension]
        for ii in range(1, len(shape)):
            temp = (cords[ii] - shape[ii] // 2) * dk[ii]
            temp *= temp
            if not self.zerocenter[ii]:
                temp = np.fft.ifftshift(temp)
            dists = dists + temp
        dists = np.sqrt(dists)
        return dists

    def get_fft_smoothing_kernel_function(self, sigma):
        return lambda x: np.exp(-2. * np.pi*np.pi * x*x * sigma*sigma)

    # ---Added properties and methods---

    @property
    def distances(self):
        """Distance between two grid points along each axis. It is a tuple
        of positive floating point numbers with the n-th entry giving the
        distances of grid points along the n-th dimension.
        """

        return self._distances

    @property
    def zerocenter(self):
        """Returns True if grid points lie symmetrically around zero.

        Returns
        -------
        bool
            True if the grid points are centered around the 0 grid point. This
            option is most common for harmonic spaces (where both conventions
            are used) but may be used for position spaces, too.

        """

        return self._zerocenter

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

    def _parse_zerocenter(self, zerocenter):
        temp = np.empty(len(self.shape), dtype=bool)
        temp[:] = zerocenter
        return tuple(temp)

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['shape'] = self.shape
        hdf5_group['zerocenter'] = self.zerocenter
        hdf5_group['distances'] = self.distances
        hdf5_group['harmonic'] = self.harmonic

        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls(
            shape=hdf5_group['shape'][:],
            zerocenter=hdf5_group['zerocenter'][:],
            distances=hdf5_group['distances'][:],
            harmonic=hdf5_group['harmonic'][()],
            )
        return result
