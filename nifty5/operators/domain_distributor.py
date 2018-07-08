import numpy as np
from ..field import Field

from .. import dobj
from ..domain_tuple import DomainTuple
from .linear_operator import LinearOperator


# MR FIXME: this needs to be rewritten in a generic fashion
class DomainDistributor(LinearOperator):
    def __init__(self, target, axis):
        if dobj.ntask > 1:
            raise NotImplementedError('UpProj class does not support MPI.')
        assert len(target) == 2
        assert axis in [0, 1]

        if axis == 0:
            domain = target[1]
            self._size = target[0].size
        else:
            domain = target[0]
            self._size = target[1].size

        self._axis = axis
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
        if mode == self.TIMES:
            x = x.local_data
            otherDirection = np.ones(self._size)
            if self._axis == 0:
                res = np.outer(otherDirection, x)
            else:
                res = np.outer(x, otherDirection)
            res = res.reshape(dobj.local_shape(self.target.shape))
            return Field.from_local_data(self.target, res)
        else:
            if self._axis == 0:
                x = x.local_data.reshape(self._size, -1)
                res = np.sum(x, axis=0)
            else:
                x = x.local_data.reshape(-1, self._size)
                res = np.sum(x, axis=1)
            res = res.reshape(dobj.local_shape(self.domain.shape))
            return Field.from_local_data(self.domain, res)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
