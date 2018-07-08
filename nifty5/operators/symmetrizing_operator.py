from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator
from .. import dobj


# MR FIXME: we should make sure that the domain is a harmonic RGSpace, correct?
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
            tmp2 = dobj.local_data(tmp)
            tmp2[lead+(slice(1, None),)] -= tmp2[lead+(slice(None, 0, -1),)]
            if i == ax:
                tmp = dobj.redistribute(tmp, dist=ax)
            return Field(self.target, val=tmp)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
