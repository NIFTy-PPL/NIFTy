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
    domain : DomainObject, i.e. Space or FieldType
        The Space on which the operator acts. The SmoothingOperator
        can only live on one space or FieldType
    sigma : float
        Sets the length of the Gaussian convolution kernel
    log_distances : boolean *optional*
        States whether the convolution happens on the logarithmic grid or not
        (default: False).
    default_spaces : tuple of ints *optional*
        Defines on which space(s) of a given field the Operator acts by
        default (default: None).

    Attributes
    ----------
    domain : DomainObject, i.e. Space or FieldType
        The domain on which the Operator's input Field lives.
    target : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain in which the outcome of the operator lives. As the Operator
        is endomorphic this is the same as its domain.
    unitary : boolean
        Indicates whether the Operator is unitary or not.
    self_adjoint : boolean
        Indicates whether the operator is self_adjoint or not.
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

    """

    # ---Overwritten properties and methods---
    def __init__(self, domain, sigma, log_distances=False,
                 default_spaces=None):
        super(SmoothingOperator, self).__init__(default_spaces)

        self._domain = self._parse_domain(domain)
        if len(self._domain) != 1:
            raise ValueError("SmoothingOperator only accepts exactly one "
                             "space as input domain.")

        self._sigma = sigma
        self._log_distances = log_distances

    def _times(self, x, spaces):
        if self.sigma == 0:
            return x.copy()

        # the domain of the smoothing operator contains exactly one space.
        # Hence, if spaces is None, but we passed LinearOperator's
        # _check_input_compatibility, we know that x is also solely defined
        # on that space
        if spaces is None:
            spaces = (0,)

        return self._smooth(x, spaces)

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

    @property
    def log_distances(self):
        return self._log_distances

    @abc.abstractmethod
    def _smooth(self, x, spaces):
        raise NotImplementedError
