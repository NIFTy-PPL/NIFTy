from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


class SymmetrizingOperator(EndomorphicOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._ndim = len(self.domain.shape)

    @property
    def domain(self):
        return self._domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        # FIXME Not efficient with MPI
        tmp = x.to_global_data().copy()
        for i in range(self._ndim):
            lead = (slice(None),)*i
            tmp[lead + (slice(1, None),)] -= tmp[lead + (slice(None, 0, -1),)]
        return Field.from_global_data(self.target, tmp)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
