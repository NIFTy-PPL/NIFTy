import numpy as np
import itertools

from .. import utilities
from .linear_operator import LinearOperator
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..field import Field

class CentralZeroPadder(LinearOperator):

    def __init__(self, domain, new_shape, space=0):
        super(CentralZeroPadder, self).__init__()

        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)
        dom = self._domain[self._space]

        if not isinstance(dom, RGSpace):
            raise TypeError("RGSpace required")
        if dom.harmonic:
            raise TypeError("RGSpace must not be harmonic")
        if len(new_shape) != len(dom.shape):
            raise ValueError("Shape missmatch")
        if any( [a<b for a,b in zip(new_shape, dom.shape)]):
            raise ValueError("New shape must be larger than old shape")

        tgt = RGSpace(new_shape, dom.distances)
        self._target = list(self._domain)
        self._target[self._space] = tgt
        self._target = DomainTuple.make(self._target)

        slicer = []
        axes = self._target.axes[self._space]
        for i in range(len(self._domain.shape)):
            if i in axes:
                slicer_fw = slice(0, self._domain.shape[i]/2)
                slicer_bw = slice(-self._domain.shape[i]/2, None)
                slicer.append( [slicer_fw, slicer_bw ] )
        self.slicer = list(itertools.product(*slicer))

        for i in range(len(self.slicer)):
            for j in range(len(self._domain.shape)):
                if not j in axes:
                    tmp = (list(self.slicer[i]))
                    tmp.insert(j, slice(None))
                    self.slicer[i] = tmp
              
    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        
        if mode == self.TIMES:
            y = np.zeros( self._target.shape )
            for i in self.slicer:
                y[i] = x[i]
            return Field(self._target, val=y)
        
        if mode == self.ADJOINT_TIMES:
            y = np.zeros( self._domain.shape )
            for i in self.slicer:
                y[i] = x[i]
            return Field(self._domain, val=y)    
