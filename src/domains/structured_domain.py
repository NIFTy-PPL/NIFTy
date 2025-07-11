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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .domain import Domain


class StructuredDomain(Domain):
    """The abstract base class for all structured domains.

    An instance of a space contains information about the manifold's
    geometry and enhances the functionality of Domain by methods that
    are needed for power spectrum analysis and smoothing.
    """

    @property
    def scalar_dvol(self):
        """float or None : uniform pixel volume, if applicable

        Returns the volume factors of this domain as a floating
        point scalar, if the volume factors are all identical, otherwise
        returns None.
        """
        raise NotImplementedError

    @property
    def dvol(self):
        """float or numpy.ndarray(dtype=float): pixel volume(s)

        Returns the volume factors of this domain, either as a floating
        point scalar (if the volume factors are all identical) or as a
        floating point array with a shape of `self.shape`.
        """
        return self.scalar_dvol

    @property
    def total_volume(self):
        """float : Total domain volume.

        Returns the sum over all the domain's pixel volumes.
        """
        tmp = self.dvol
        return self.size * tmp if np.isscalar(tmp) else np.sum(tmp)

    @property
    def harmonic(self):
        """bool : True iff this domain is a harmonic domain."""
        raise NotImplementedError

    def get_k_length_array(self):
        """k vector lengths, if applicable.

        Returns the length of the k vector for every pixel.
        This method is only implemented for harmonic domains.

        Returns
        -------
        :class:`nifty8.field.Field`
            An array containing the k vector lengths
        """
        raise NotImplementedError

    def get_unique_k_lengths(self):
        """Sorted unique k-vector lengths, if applicable.

        Returns an array of floats containing the unique k vector lengths
        for this domain.
        This method is only implemented for harmonic domains.
        """
        raise NotImplementedError

    def get_fft_smoothing_kernel_function(self, sigma):
        """Helper for Gaussian smoothing.

        This method, which is only implemented for harmonic domains, helps to
        smoothe fields that are defined on a domain that has this domain as
        its harmonic partner. The returned function does a pointwise evaluation
        of a zero-centered Gaussian on the field values, which corresponds to a
        convolution with a Gaussian kernel with sigma standard deviation in
        position space.

        Parameters
        ----------
        sigma : float
            A real number representing a physical scale on which the smoothing
            takes place. Mathematically sigma is the standard
            deviation of a convolution with a normalized, zero-centered
            Gaussian that takes place in position space.

        Returns
        -------
        function (array-like -> array-like)
            A smoothing operation that multiplies values with a Gaussian
            kernel.
        """
        raise NotImplementedError
