from __future__ import absolute_import, division, print_function

import numpy as np

from .. import dobj
from ..compat import *
from ..field import Field
from ..domains.rg_space import RGSpace
from ..domain_tuple import DomainTuple
from .linear_operator import LinearOperator


class FieldZeroPadder(LinearOperator):
    def __init__(self, target, factor, space=0):
        super(FieldZeroPadder, self).__init__()
        self._target = DomainTuple.make(target)
        self._space = int(space)
        tgt = self._target[self._space]
        if not isinstance(tgt, RGSpace):
            raise TypeError("RGSpace required")
        if not len(tgt.shape) == 1:
            raise TypeError("RGSpace must be one-dimensional")
        if tgt.harmonic:
            raise TypeError("RGSpace must not be harmonic")

        dom = RGSpace((int(factor*tgt.shape[0]),), tgt.distances)
        self._domain = list(self._target)
        self._domain[self._space] = dom
        self._domain = DomainTuple.make(self._domain)

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
        dax = dobj.distaxis(x)
        shp_in = x.shape
        shp_out = self._tgt(mode).shape
        ax = self._domain.axes[self._space][0]
        if dax == ax:
            x = dobj.redistribute(x, nodist=(ax,))
        curax = dobj.distaxis(x)

        if mode == self.TIMES:
            newarr = np.empty(dobj.local_shape(shp_out), dtype=x.dtype)
            newarr[()] = dobj.local_data(x)[(slice(None),)*ax +
                                            (slice(0, shp_out[ax]),)]
        else:
            newarr = np.zeros(dobj.local_shape(shp_out), dtype=x.dtype)
            newarr[(slice(None),)*ax +
                   (slice(0, shp_in[ax]),)] = dobj.local_data(x)
        newarr = dobj.from_local_data(shp_out, newarr, distaxis=curax)
        if dax == ax:
            newarr = dobj.redistribute(newarr, dist=ax)
        return Field(self._tgt(mode), val=newarr)
