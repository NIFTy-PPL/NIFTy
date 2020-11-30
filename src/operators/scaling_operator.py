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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .endomorphic_operator import EndomorphicOperator


class ScalingOperator(EndomorphicOperator):
    """Operator which multiplies a Field with a scalar.

    Parameters
    ----------
    domain : Domain or tuple of Domain or DomainTuple
        The domain on which the Operator's input Field is defined.
    factor : scalar
        The multiplication factor

    Notes
    -----
    :class:`Operator` supports the multiplication with a scalar. So one does
    not need instantiate :class:`ScalingOperator` explicitly in most cases.

    Formally, this operator always supports all operation modes (times,
    adjoint_times, inverse_times and inverse_adjoint_times), even if `factor`
    is 0 or infinity. It is the user's responsibility to apply the operator
    only in appropriate ways (e.g. call inverse_times only if `factor` is
    nonzero).

    Along with this behaviour comes the feature that it is possible to draw an
    inverse sample from a :class:`ScalingOperator` (which is a zero-field).
    This occurs if one draws an inverse sample of a positive definite sum of
    two operators each of which are only positive semi-definite. However, it
    is unclear whether this beviour does not lead to unwanted effects
    somewhere else.
    """

    def __init__(self, domain, factor):
        from ..sugar import makeDomain

        if not np.isscalar(factor):
            raise TypeError("Scalar required")
        self._domain = makeDomain(domain)
        self._factor = factor
        self._capability = self._all_ops

    def apply(self, x, mode):
        from ..sugar import full

        self._check_input(x, mode)
        fct = self._factor
        if fct == 1.:
            return x
        if fct == 0.:
            return full(x.domain, 0.)

        MODES_WITH_ADJOINT = self.ADJOINT_TIMES | self.ADJOINT_INVERSE_TIMES
        MODES_WITH_INVERSE = self.INVERSE_TIMES | self.ADJOINT_INVERSE_TIMES
        if (mode & MODES_WITH_ADJOINT) != 0:
            fct = np.conj(fct)
        if (mode & MODES_WITH_INVERSE) != 0:
            fct = 1./fct
        return x*fct

    def _flip_modes(self, trafo):
        fct = self._factor
        if trafo & self.ADJOINT_BIT:
            fct = np.conj(fct)
        if trafo & self.INVERSE_BIT:
            fct = 1./fct
        return ScalingOperator(self._domain, fct)

    def _get_fct(self, from_inverse):
        fct = self._factor
        if (fct.imag != 0. or fct.real < 0. or
                (fct.real == 0. and from_inverse)):
            raise ValueError("operator not positive definite")
        return 1./np.sqrt(fct) if from_inverse else np.sqrt(fct)

    def draw_sample_with_dtype(self, dtype, from_inverse=False):
        from ..sugar import from_random
        return from_random(domain=self._domain, random_type="normal", dtype=dtype, std=self._get_fct(from_inverse))

    def get_sqrt(self):
        fct = self._get_fct(False)
        if np.iscomplexobj(fct) or fct < 0:
            raise ValueError("get_sqrt() works only for positive definite operators.")
        return ScalingOperator(self._domain, fct)

    def __call__(self, other):
        res = EndomorphicOperator.__call__(self, other)
        if np.isreal(self._factor) and self._factor >= 0:
            if other.jac is not None:
                if other.metric is not None:
                    from .sandwich_operator import SandwichOperator
                    sqrt_fac = np.sqrt(self._factor)
                    newop = ScalingOperator(other.metric.domain, sqrt_fac)
                    met = SandwichOperator.make(newop, other.metric)
                    res = res.add_metric(met)
        return res

    def __repr__(self):
        return "ScalingOperator ({})".format(self._factor)
