import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


class OuterOperator(EndomorphicOperator):
    def __init__(self, field):
        super(OuterOperator, self).__init__()
        if not isinstance(field, Field):
            raise TypeError("Field object required")
        self._field = field
        self._domain = DomainTuple.make(field.domain)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            fac = np.sum(x.to_global_data())
            return fac * self._field
        else:
            fac = self._field.vdot(x)
            return fac * Field.ones(self.domain)

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
