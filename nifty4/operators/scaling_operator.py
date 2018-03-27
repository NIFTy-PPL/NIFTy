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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
import numpy as np
from ..field import Field
from ..domain_tuple import DomainTuple
from .endomorphic_operator import EndomorphicOperator


class ScalingOperator(EndomorphicOperator):
    """Operator which multiplies a Field with a scalar.

    The NIFTy ScalingOperator class is a subclass derived from the
    EndomorphicOperator. It multiplies an input field with a given factor.

    Parameters
    ----------
    factor : scalar
        The multiplication factor
    domain : Domain or tuple of Domain or DomainTuple
        The domain on which the Operator's input Field lives.

    Notes
    -----
    Formally, this operator always supports all operation modes (times,
    adjoint_times, inverse_times and inverse_adjoint_times), even if `factor`
    is 0 or infinity. It is the user's responsibility to apply the operator
    only in appropriate ways (e.g. call inverse_times only if `factor` is
    nonzero).

    This shortcoming will hopefully be fixed in the future.
    """

    def __init__(self, factor, domain):
        super(ScalingOperator, self).__init__()

        if not np.isscalar(factor):
            raise TypeError("Scalar required")
        self._factor = factor
        self._domain = DomainTuple.make(domain)

    def apply(self, x, mode):
        self._check_input(x, mode)

        if self._factor == 1.:
            return x.copy()
        if self._factor == 0.:
            return Field.zeros_like(x, dtype=x.dtype)

        if mode == self.TIMES:
            return x*self._factor
        elif mode == self.ADJOINT_TIMES:
            return x*np.conj(self._factor)
        elif mode == self.INVERSE_TIMES:
            return x*(1./self._factor)
        else:
            return x*(1./np.conj(self._factor))

    @property
    def inverse(self):
        if self._factor != 0.:
            return ScalingOperator(1./self._factor, self._domain)
        from .inverse_operator import InverseOperator
        return InverseOperator(self)

    @property
    def adjoint(self):
        return ScalingOperator(np.conj(self._factor), self._domain)

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        if self._factor == 0.:
            return self.TIMES | self.ADJOINT_TIMES
        return self._all_ops

    def process_sample(self, sample):
        if self._factor.imag != 0. or self._factor.real <= 0.:
            raise ValueError("Operator not positive definite")
        return sample * np.sqrt(self._factor)

    def _sample_helper(self, fct, dtype):
        if fct.imag != 0. or fct.real <= 0.:
            raise ValueError("operator not positive definite")
        return Field.from_random(
           random_type="normal", domain=self._domain, std=fct, dtype=dtype)

    def draw_sample(self, dtype=np.float64):
        return self._sample_helper(np.sqrt(self._factor), dtype)

    def inverse_draw_sample(self, dtype=np.float64):
        return self._sample_helper(1./np.sqrt(self._factor), dtype)
