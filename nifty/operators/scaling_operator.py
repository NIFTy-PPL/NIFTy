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

from __future__ import division
import numpy as np
from ..field import Field
from ..domain_tuple import DomainTuple
from .endomorphic_operator import EndomorphicOperator
from .. import dobj


class ScalingOperator(EndomorphicOperator):
    """ NIFTY class for an operator which multiplies a Field with a scalar.

    The NIFTY ScalingOperator class is a subclass derived from the
    EndomorphicOperator. It multiplies an input field with a given factor.

    Parameters
    ----------
    factor : scalar
        The multiplication factor
    domain : tuple of DomainObjects
        The domain on which the Operator's input Field lives.

    Attributes
    ----------
    domain : DomainTuple
        The domain on which the Operator's input Field lives.
    """

    def __init__(self, factor, domain):
        super(ScalingOperator, self).__init__()

        if not np.isscalar(factor):
            raise TypeError("Scalar required")
        self._factor = factor
        self._domain = DomainTuple.make(domain)

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            return x*self._factor
        elif mode == self.ADJOINT_TIMES:
            return x*np.conj(self._factor)
        elif mode == self.INVERSE_TIMES:
            return x*(1./self._factor)
        else:
            return x*(1./np.conj(self._factor))

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return (self.TIMES | self.ADJOINT_TIMES |
                self.INVERSE_TIMES | self.ADJOINT_INVERSE_TIMES)
