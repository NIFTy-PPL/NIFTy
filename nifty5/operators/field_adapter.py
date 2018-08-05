from __future__ import absolute_import, division, print_function

from ..compat import *
from .linear_operator import LinearOperator
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..field import Field


class FieldAdapter(LinearOperator):
    def __init__(self, dom, name_dom):
        self._domain = MultiDomain.make(dom)
        self._name = name_dom
        self._target = dom[name_dom]

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

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
