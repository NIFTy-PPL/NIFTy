from ..domain_tuple import DomainTuple
from ..field import Field
from .. import dobj
from ..utilities import hartley
from .linear_operator import LinearOperator


class QHTOperator(LinearOperator):
    def __init__(self, domain, target):
        if not domain.harmonic:
            raise TypeError(
                "HarmonicTransformOperator only works on a harmonic space")
        if target.harmonic:
            raise TypeError("Target is not a codomain of domain")

        from ..domains import LogRGSpace
        if not isinstance(domain, LogRGSpace):
            raise ValueError("Domain has to be a LogRGSpace!")
        if not isinstance(target, LogRGSpace):
            raise ValueError("Target has to be a LogRGSpace!")

        self._domain = DomainTuple.make(domain)
        self._target = DomainTuple.make(target)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val * self.domain[0].scalar_dvol()
        n = len(self.domain[0].shape)
        rng = range(n) if mode == self.TIMES else reversed(range(n))
        for i in rng:
            sl = (slice(None),)*i + (slice(1, None),)
            if i == dobj.distaxis(x):
                x = dobj.redistribute(x, nodist=(i,))
                ax = dobj.distaxis(x)
                x = dobj.local_data(x)
                x[sl] = hartley(x[sl], axes=(i,))
                x = dobj.from_local_data(x.shape, x, distaxis=ax)
                x = dobj.redistribute(x, dist=i)
            else:
                x[sl] = hartley(x[sl], axes=(i,))

        return Field(self._tgt(mode), val=x)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
