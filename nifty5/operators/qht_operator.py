from ..domain_tuple import DomainTuple
from ..field import Field
from .. import dobj
from ..utilities import hartley
from .linear_operator import LinearOperator


class QHTOperator(LinearOperator):
    """
    Does a Hartley transform on LogRGSpace

    This operator takes a field on a LogRGSpace and transforms it
    according to the Hartley transform. The zero modes are not transformed
    because they are infinitely far away.

    Parameters
    ----------
    domain : LogRGSpace
        The domain needs to be a LogRGSpace.
    target : LogRGSpace
        The target needs to be a LogRGSpace.
    """
    def __init__(self, domain, target):
        if not domain.harmonic:
            raise TypeError(
                "HarmonicTransformOperator only works on a harmonic space")
        if target.harmonic:
            raise TypeError("Target is not a codomain of domain")

        from ..domains.log_rg_space import LogRGSpace
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
        ax = dobj.distaxis(x)
        globshape = x.shape
        for i in rng:
            sl = (slice(None),)*i + (slice(1, None),)
            if i == ax:
                x = dobj.redistribute(x, nodist=(ax,))
            curax = dobj.distaxis(x)
            x = dobj.local_data(x)
            x[sl] = hartley(x[sl], axes=(i,))
            x = dobj.from_local_data(globshape, x, distaxis=curax)
            if i == ax:
                x = dobj.redistribute(x, dist=ax)
        return Field(self._tgt(mode), val=x)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
