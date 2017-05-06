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

import abc

from nifty.domain_object import DomainObject


class Space(DomainObject):
    """The abstract base class for all NIFTy spaces.

    An instance of a space contains information about the manifolds geometry
    and enhances the functionality of DomainObject by methods that are needed
    for powerspectrum analysis and smoothing.

            Parameters
            ----------
    None

    Attributes
    ----------
    dim : np.int
        Total number of dimensionality, i.e. the number of pixels.
    harmonic : bool
        Specifies whether the space is a signal or harmonic space.
    total_volume : np.float
        The total volume of the space.
    shape : tuple of np.ints
        The shape of the space's data array.

    Raises
    ------
    TypeError
        Raised if instantiated directly.

    Notes
    -----
    `Space` is an abstract base class. In order to allow for instantiation the
    methods `get_distance_array`, `total_volume` and `copy` must be implemented
    as well as the abstract methods inherited from `DomainObject`.

    See Also
    --------
    distributor
        """
    def __init__(self):
        super(Space, self).__init__()

    @abc.abstractproperty
    def harmonic(self):
        raise NotImplementedError

    @abc.abstractproperty
    def total_volume(self):
        raise NotImplementedError(
            "There is no generic volume for the Space base class.")

    @abc.abstractmethod
    def copy(self):
        return self.__class__()

    def get_distance_array(self, distribution_strategy):
        raise NotImplementedError(
            "There is no generic distance structure for Space base class.")

    def get_fft_smoothing_kernel_function(self, sigma):
        raise NotImplementedError(
            "There is no generic co-smoothing kernel for Space base class.")

    def hermitian_decomposition(self, x, axes=None,
                                preserve_gaussian_variance=False):
        raise NotImplementedError

    def __repr__(self):
        string = ""
        string += str(type(self)) + "\n"
        return string
