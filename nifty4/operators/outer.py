from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


class OuterOperator(EndomorphicOperator):
    def __init__(self, a, b):
        super(OuterOperator, self).__init__()
        if not isinstance(a, Field) or not isinstance(b, Field):
            raise TypeError("Field object required")
        assert a.domain == b.domain
        self._a = a
        self._b = b
        self._domain = DomainTuple.make(a.domain)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return self._a * self._b.vdot(x)
        else:
            return self._b * self._a.vdot(x)

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
