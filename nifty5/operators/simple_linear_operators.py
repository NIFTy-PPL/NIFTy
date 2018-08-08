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

from __future__ import absolute_import, division, print_function

import numpy as np

from ..compat import *
from ..domain_tuple import DomainTuple
from ..multi_domain import MultiDomain
from ..domains.unstructured_domain import UnstructuredDomain
from .linear_operator import LinearOperator
from .endomorphic_operator import EndomorphicOperator
from ..sugar import full
from ..field import Field
from ..multi_field import MultiField


class VdotOperator(LinearOperator):
    def __init__(self, field):
        super(VdotOperator, self).__init__()
        self._field = field
        self._domain = field.domain
        self._target = DomainTuple.scalar_domain()

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field(self._target, self._field.vdot(x))
        return self._field*x.local_data[()]


class SumReductionOperator(LinearOperator):
    def __init__(self, domain):
        super(SumReductionOperator, self).__init__()
        self._domain = domain
        self._target = DomainTuple.scalar_domain()

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field(self._target, x.sum())
        return full(self._domain, x.local_data[()])


class ConjugationOperator(EndomorphicOperator):
    def __init__(self, domain):
        super(ConjugationOperator, self).__init__()
        self._domain = domain

    @property
    def capability(self):
        return self._all_ops

    def apply(self, x, mode):
        self._check_input(x, mode)
        return x.conjugate()


class Realizer(EndomorphicOperator):
    def __init__(self, domain):
        super(Realizer, self).__init__()
        self._domain = domain

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return x.real


class FieldAdapter(LinearOperator):
    def __init__(self, dom, name_dom):
        self._domain = MultiDomain.make(dom)
        self._name = name_dom
        self._target = dom[name_dom]

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            return x[self._name]
        values = tuple(Field.full(dom, 0.) if key != self._name else x
                       for key, dom in self._domain.items())
        return MultiField(self._domain, values)


class GeometryRemover(LinearOperator):
    """Operator which transforms between a structured and an unstructured
    domain.

    Parameters
    ----------
    domain: Domain, tuple of Domain, or DomainTuple:
        the full input domain of the operator.

    Notes
    -----
    The operator will convert every sub-domain of its input domain to an
    UnstructuredDomain with the same shape. No weighting by volume factors
    is carried out.
    """

    def __init__(self, domain):
        super(GeometryRemover, self).__init__()
        self._domain = DomainTuple.make(domain)
        target_list = [UnstructuredDomain(dom.shape) for dom in self._domain]
        self._target = DomainTuple.make(target_list)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return x.cast_domain(self._target)
        return x.cast_domain(self._domain)


class NullOperator(LinearOperator):
    """Operator corresponding to a matrix of all zeros.

    Parameters
    ----------
    domain : DomainTuple or MultiDomain
        input domain
    target : DomainTuple or MultiDomain
        output domain
    """

    def __init__(self, domain, target):
        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)

    @staticmethod
    def _nullfield(dom):
        if isinstance(dom, DomainTuple):
            return Field.full(dom, 0)
        else:
            return MultiField.full(dom, 0)

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            return self._nullfield(self._target)
        return self._nullfield(self._domain)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
