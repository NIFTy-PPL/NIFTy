from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


class RowOperator(EndomorphicOperator):
    def __init__(self, field):
        super(RowOperator, self).__init__()
        if not isinstance(field, Field):
            raise TypeError("Field object required")

        self._field = field
        self._domain = DomainTuple.make(field.domain)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field.full(self.target, self._field.vdot(x))
        else:
            return self._field * x.sum()

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
