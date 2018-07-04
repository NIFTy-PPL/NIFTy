from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator
from .. import dobj


class SymmetrizingOperator(EndomorphicOperator):
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._ndim = len(self.domain.shape)

    @property
    def domain(self):
        return self._domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        tmp = x.val.copy()
        ax = dobj.distaxis(tmp)
        globshape = tmp.shape
        for i in range(self._ndim):
            lead = (slice(None),)*i
            if i == ax:
                tmp = dobj.redistribute(tmp, nodist=(ax,))
            curax = dobj.distaxis(tmp)
            tmp = dobj.local_data(tmp)
            tmp[lead + (slice(1, None),)] -= tmp[lead + (slice(None, 0, -1),)]
            tmp = dobj.from_local_data(globshape, tmp, distaxis=curax)
            if i == ax:
                tmp = dobj.redistribute(tmp, dist=ax)
            return Field(self.target, val=tmp)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
