import numpy as np
from functools import reduce
from ..domains.rg_space import RGSpace
from ..domains.irg_space import IRGSpace
from .endomorphic_operator import EndomorphicOperator
from ..sugar import makeDomain, makeField

class CumsumOperator(EndomorphicOperator):
    def __init__(self, domain, spaces = None):
        self._domain = makeDomain(domain)
        if spaces is None:
            spaces  = 0
        intdom = self._domain[spaces]
        if isinstance(intdom, RGSpace):
            if not len(intdom.distances) == 1:
                raise ValueError("Integration domain must be 1D")
            self._wgts = np.ones(intdom.shape)*intdom.distances[0]
        elif isinstance(intdom, IRGSpace):
            self._wgts = intdom.dvol
        else:
            raise ValueError("Integration domain of incorrect type!")

        self._axis = reduce(lambda a,b:a+b, (len(dd.shape) for dd in 
                                            self._domain[:spaces]))
        _back = reduce(lambda a,b:a+b, (len(dd.shape) for dd in 
                                       self._domain[(spaces+1):]))
        self._wgts = np.expand_dims(self._wgts, 
                                    axis=tuple(i for i in range(self._axis)) + 
                                    tuple(-(i+1) for i in range(_back)))

        self._capabilities = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = self._wgts*x.val
        if mode == self.ADJOINT_TIMES:
            x = np.flip(x, axis=self._axis)
        res = np.cumsum(x, axis=self._axis)
        if mode == self.ADJOINT_TIMES:
            res = np.flip(res, axis=self._axis)
        return makeField(self._domain, res)