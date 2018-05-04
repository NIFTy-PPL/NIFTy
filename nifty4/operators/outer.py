from ..field import Field
from .endomorphic_operator import EndomorphicOperator
from .row_operator import RowOperator


class OuterOperator(EndomorphicOperator):
    def __init__(self, a, b):
        super(OuterOperator, self).__init__()
        if not isinstance(a, Field) or not isinstance(b, RowOperator):
            raise TypeError("Field object and RowOperator required")
        assert a.domain == b.domain
        self._a = a
        self._b = b

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            fac = float(self._b(x).to_global_data())
            return self._a * fac
        else:
            return self._b.adjoint(Field.full(self._b.target, self._a.vdot(x)))

    @property
    def domain(self):
        return self._a.domain

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
