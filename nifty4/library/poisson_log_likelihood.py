from numpy import inf, isnan

from ..minimization import Energy
from ..operators import SandwichOperator
from ..sugar import log, makeOp


class PoissonLogLikelihood(Energy):
    def __init__(self, position, lamb, d):
        """
        s: Sky model object

        value = 0.5 * s.vdot(s), i.e. a log-Gauss distribution with unit
        covariance
        """
        super(PoissonLogLikelihood, self).__init__(position)
        self._lamb = lamb.at(position)
        self._d = d

        lamb_val = self._lamb.value

        self._value = lamb_val.sum() - d.vdot(log(lamb_val))
        if isnan(self._value):
            self._value = inf
        self._gradient = self._lamb.gradient.adjoint_times(1 - d/lamb_val)

        # metric = makeOp(d/lamb_val/lamb_val)
        metric = makeOp(1./lamb_val)
        self._curvature = SandwichOperator.make(self._lamb.gradient, metric)

    def at(self, position):
        return self.__class__(position, self._lamb, self._d)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    def curvature(self):
        return self._curvature
