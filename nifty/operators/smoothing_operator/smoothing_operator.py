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

import numpy as np

from nifty.operators.endomorphic_operator import EndomorphicOperator
from nifty.spaces import RGSpace, GLSpace, HPSpace, PowerSpace


class SmoothingOperator(EndomorphicOperator):
    """ NIFTY class for smoothing operators.

    The NIFTy SmoothingOperator smooths Fields, with a given kernel length.
    Fields which are not living over a PowerSpace are smoothed
    via a gaussian convolution. Fields living over the PowerSpace are directly
    smoothed.

    Parameters
    ----------
    domain : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The Space on which the operator acts
    sigma : float
        Sets the length of the Gaussian convolution kernel
    log_distances : boolean
        States whether the convolution happens on the logarithmic grid or not.

    Attributes
    ----------
    sigma : float
        Sets the length of the Gaussian convolution kernel
    log_distances : boolean
        States whether the convolution happens on the logarithmic grid or not.

    Raises
    ------
    ValueError
        Raised if
            * the given domain inherits more than one space. The
              SmoothingOperator acts only on one Space.

    Notes
    -----

    Examples
    --------
    >>> x = RGSpace(5)
    >>> S = SmoothingOperator(x, sigma=1.)
    >>> f = Field(x, val=[1,2,3,4,5])
    >>> S.times(f).val
    <distributed_data_object>
    array([ 3.,  3.,  3.,  3.,  3.])

    See Also
    --------
    DiagonalOperator, SmoothingOperator,
    PropagatorOperator, ProjectionOperator,
    ComposedOperator

    """

    _fft_smoothing_spaces = [RGSpace,
                             GLSpace,
                             HPSpace]
    _direct_smoothing_spaces = [PowerSpace]

    def __new__(cls, domain, *args, **kwargs):
        if cls is SmoothingOperator:
            domain = cls._parse_domain(domain)

            if len(domain) != 1:
                raise ValueError("SmoothingOperator only accepts exactly one "
                                 "space as input domain.")

            if np.any([isinstance(domain[0], sp)
                       for sp in cls._fft_smoothing_spaces]):
                from .fft_smoothing_operator import FFTSmoothingOperator
                return super(SmoothingOperator, cls).__new__(
                        FFTSmoothingOperator, domain, *args, **kwargs)

            elif np.any([isinstance(domain[0], sp)
                         for sp in cls._direct_smoothing_spaces]):
                from .direct_smoothing_operator import DirectSmoothingOperator
                return super(SmoothingOperator, cls).__new__(
                        DirectSmoothingOperator, domain, *args, **kwargs)

            else:
                raise NotImplementedError("For the given Space smoothing "
                                          " is not available.")
        else:
            print 'new 4'
            return super(SmoothingOperator, cls).__new__(cls,
                                                         domain,
                                                         *args,
                                                         **kwargs)

    # ---Overwritten properties and methods---
    def __init__(self, domain, sigma, log_distances=False,
                 default_spaces=None):
        super(SmoothingOperator, self).__init__(default_spaces)

        # # the _parse_domain is already done in the __new__ method
        # self._domain = self._parse_domain(domain)
        # if len(self.domain) != 1:
        #     raise ValueError("SmoothingOperator only accepts exactly one "
        #                      "space as input domain.")
        self._domain = self._parse_domain(domain)

        self.sigma = sigma
        self.log_distances = log_distances

    def _inverse_times(self, x, spaces):
        if self.sigma == 0:
            return x.copy()

        # the domain of the smoothing operator contains exactly one space.
        # Hence, if spaces is None, but we passed LinearOperator's
        # _check_input_compatibility, we know that x is also solely defined
        # on that space
        if spaces is None:
            spaces = (0,)

        return self._smooth(x, spaces, inverse=True)

    def _times(self, x, spaces):
        if self.sigma == 0:
            return x.copy()

        # the domain of the smoothing operator contains exactly one space.
        # Hence, if spaces is None, but we passed LinearOperator's
        # _check_input_compatibility, we know that x is also solely defined
        # on that space
        if spaces is None:
            spaces = (0,)

        return self._smooth(x, spaces, inverse=False)

    # ---Mandatory properties and methods---
    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    # ---Added properties and methods---

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = np.float(sigma)

    @property
    def log_distances(self):
        return self._log_distances

    @log_distances.setter
    def log_distances(self, log_distances):
        self._log_distances = bool(log_distances)

    @abc.abstractmethod
    def _smooth(self, x, spaces, inverse):
        raise NotImplementedError
