from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import LinearOperator


class RowOperator(LinearOperator):
    def __init__(self, field):
        super(RowOperator, self).__init__()
        if not isinstance(field, Field):
            raise TypeError("Field object required")

        self._field = field
        self._domain = DomainTuple.make(field.domain)
        self._target = DomainTuple.make(())

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field(self.target, self._field.vdot(x))
        else:
            if len(x.domain) == 0:
                x = x.val[()]
            return self._field * x

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
