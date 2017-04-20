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

import numpy as np

from d2o import distributed_data_object,\
                STRATEGIES as DISTRIBUTION_STRATEGIES

from nifty.spaces.space import Space


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
        num : {int, numpy.ndarray}
            Number of gridpoints or numbers of gridpoints along each axis.
        naxes : int, *optional*
            Number of axes (default: None).
        zerocenter : {bool, numpy.ndarray}, *optional*
            Whether the Fourier zero-mode is located in the center of the grid
            (or the center of each axis speparately) or not (default: True).
        hermitian : bool, *optional*
            Whether the fields living in the space follow hermitian symmetry or
            not (default: True).
        purelyreal : bool, *optional*
            Whether the field values are purely real (default: True).
        dist : {float, numpy.ndarray}, *optional*
            Distance between two grid points along each axis (default: None).
        fourier : bool, *optional*
            Whether the space represents a Fourier or a position grid
            (default: False).

        Notes
        -----
        Only even numbers of grid points per axis are supported.
        The basis transformations between position `x` and Fourier mode `k`
        rely on (inverse) fast Fourier transformations using the
        :math:`exp(2 \pi i k^\dagger x)`-formulation.

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing information on the axes of the
            space in the following form: The first entries give the grid-points
            along each axis in reverse order; the next entry is 0 if the
            fields defined on the space are purely real-valued, 1 if they are
            hermitian and complex, and 2 if they are not hermitian, but
            complex-valued; the last entries hold the information on whether
            the axes are centered on zero or not, containing a one for each
            zero-centered axis and a zero for each other one, in reverse order.
        dtype : numpy.dtype
            Data type of the field values for a field defined on this space,
            either ``numpy.float64`` or ``numpy.complex128``.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for regular grids.
        vol : numpy.ndarray
            One-dimensional array containing the distances between two grid
            points along each axis, in reverse order. By default, the total
            length of each axis is assumed to be one.
        fourier : bool
            Whether or not the grid represents a Fourier basis.
    """

    # ---Overwritten properties and methods---

    def __init__(self, shape=(1,), zerocenter=False, distances=None,
                 harmonic=False, dtype=None):
        """
            Sets the attributes for an rg_space class instance.

            Parameters
            ----------
            num : {int, numpy.ndarray}
                Number of gridpoints or numbers of gridpoints along each axis.
            naxes : int, *optional*
                Number of axes (default: None).
            zerocenter : {bool, numpy.ndarray}, *optional*
                Whether the Fourier zero-mode is located in the center of the
                grid (or the center of each axis speparately) or not
                (default: False).
            hermitian : bool, *optional*
                Whether the fields living in the space follow hermitian
                symmetry or not (default: True).
            purelyreal : bool, *optional*
                Whether the field values are purely real (default: True).
            dist : {float, numpy.ndarray}, *optional*
                Distance between two grid points along each axis
                (default: None).
            fourier : bool, *optional*
                Whether the space represents a Fourier or a position grid
                (default: False).

            Returns
            -------
            None
        """
        self._harmonic = bool(harmonic)

        if dtype is None:
            if self.harmonic:
                dtype = np.dtype('complex')
            else:
                dtype = np.dtype('float')

        super(RGSpace, self).__init__(dtype)

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
        anti_hermitian_part = (x-hermitian_part)/1j

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

        # The fixed points of the point inversion must not be avaraged.
        # Hence one must divide out the sqrt(2) again
        # -> Get the middle index of the array
        mid_index = np.array(hermitian_part.shape, dtype=np.int) // 2
        dimensions = mid_index.size
        # Use ndindex to iterate over all combinations of zeros and the
        # mid_index in order to correct all fixed points.
        if axes is None:
            axes = xrange(dimensions)

        ndlist = [2 if i in axes else 1 for i in xrange(dimensions)]
        ndlist = tuple(ndlist)
        for i in np.ndindex(ndlist):
            temp_index = tuple(i * mid_index)
            hermitian_part[temp_index] /= np.sqrt(2)
            anti_hermitian_part[temp_index] /= np.sqrt(2)
        return hermitian_part, anti_hermitian_part

    def _hermitianize_inverter(self, x, axes):
        # calculate the number of dimensions the input array has
        dimensions = len(x.shape)
        # prepare the slicing object which will be used for mirroring
        slice_primitive = [slice(None), ] * dimensions
        # copy the input data
        y = x.copy()

        if axes is None:
            axes = xrange(dimensions)

        # flip in the desired directions
        for i in axes:
            slice_picker = slice_primitive[:]
            slice_picker[i] = slice(1, None, None)
            slice_picker = tuple(slice_picker)

            slice_inverter = slice_primitive[:]
            slice_inverter[i] = slice(None, 0, -1)
            slice_inverter = tuple(slice_inverter)

            try:
                y.set_data(to_key=slice_picker, data=y,
                           from_key=slice_inverter)
            except(AttributeError):
                y[slice_picker] = y[slice_inverter]
        return y

    # ---Mandatory properties and methods---

    @property
    def harmonic(self):
        return self._harmonic

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return reduce(lambda x, y: x*y, self.shape)

    @property
    def total_volume(self):
        return self.dim * reduce(lambda x, y: x*y, self.distances)

    def copy(self):
        return self.__class__(shape=self.shape,
                              zerocenter=self.zerocenter,
                              distances=self.distances,
                              harmonic=self.harmonic,
                              dtype=self.dtype)

    def weight(self, x, power=1, axes=None, inplace=False):
        weight = reduce(lambda x, y: x*y, self.distances)**power
        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x*weight
        return result_x

    def get_distance_array(self, distribution_strategy):
        """
            Calculates an n-dimensional array with its entries being the
            lengths of the k-vectors from the zero point of the grid.

            Parameters
            ----------
            None : All information is taken from the parent object.

            Returns
            -------
            nkdict : distributed_data_object
        """
        shape = self.shape
        # prepare the distributed_data_object
        nkdict = distributed_data_object(
                        global_shape=shape,
                        distribution_strategy=distribution_strategy,
                        dtype=np.float64)

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

        dists = ((np.float128(0) + cords[0] - shape[0] // 2) * dk[0])**2
        # apply zerocenterQ shift
        if not self.zerocenter[0]:
            dists = np.fft.ifftshift(dists)
        # only save the individual slice
        dists = dists[slice_of_first_dimension]
        for ii in range(1, len(shape)):
            temp = ((cords[ii] - shape[ii] // 2) * dk[ii])**2
            if not self.zerocenter[ii]:
                temp = np.fft.fftshift(temp)
            dists = dists + temp
        dists = np.sqrt(dists)
        return dists

    def get_fft_smoothing_kernel_function(self, sigma):
        if sigma is None:
            sigma = np.sqrt(2) * np.max(self.distances)

        return lambda x: np.exp(-2. * np.pi**2 * x**2 * sigma**2)

    # ---Added properties and methods---

    @property
    def distances(self):
        return self._distances

    @property
    def zerocenter(self):
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
                temp = np.ones_like(self.shape, dtype=np.float)
            else:
                temp = 1 / np.array(self.shape, dtype=np.float)
        else:
            temp = np.empty(len(self.shape), dtype=np.float)
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
        hdf5_group.attrs['dtype'] = self.dtype.name

        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls(
            shape=hdf5_group['shape'][:],
            zerocenter=hdf5_group['zerocenter'][:],
            distances=hdf5_group['distances'][:],
            harmonic=hdf5_group['harmonic'][()],
            dtype=np.dtype(hdf5_group.attrs['dtype'])
            )
        return result
