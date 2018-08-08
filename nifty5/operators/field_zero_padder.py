from __future__ import absolute_import, division, print_function

import numpy as np

from .. import dobj
from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..field import Field
from .linear_operator import LinearOperator
from .. import utilities


class FieldZeroPadder(LinearOperator):
    def __init__(self, domain, new_shape, space=0):
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)
        dom = self._domain[self._space]
        if not isinstance(dom, RGSpace):
            raise TypeError("RGSpace required")
        if dom.harmonic:
            raise TypeError("RGSpace must not be harmonic")

        if len(new_shape) != len(dom.shape):
            raise ValueError("Shape mismatch")
        if any([a < b for a, b in zip(new_shape, dom.shape)]):
            raise ValueError("New shape must be larger than old shape")
        self._target = list(self._domain)
        self._target[self._space] = RGSpace(new_shape, dom.distances)
        self._target = DomainTuple.make(self._target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        dax = dobj.distaxis(x)
        shp_in = x.shape
        shp_out = self._tgt(mode).shape
        axbefore = self._target.axes[self._space][0]
        axes = self._target.axes[self._space]
        if dax in axes:
            x = dobj.redistribute(x, nodist=axes)
        curax = dobj.distaxis(x)

        if mode == self.ADJOINT_TIMES:
            newarr = np.empty(dobj.local_shape(shp_out, curax), dtype=x.dtype)
            sl = tuple(slice(0, shp_out[axis]) for axis in axes)
            newarr[()] = dobj.local_data(x)[(slice(None),)*axbefore + sl]
        else:
            newarr = np.zeros(dobj.local_shape(shp_out, curax), dtype=x.dtype)
            sl = tuple(slice(0, shp_in[axis]) for axis in axes)
            newarr[(slice(None),)*axbefore + sl] = dobj.local_data(x)
        newarr = dobj.from_local_data(shp_out, newarr, distaxis=curax)
        if dax in axes:
            newarr = dobj.redistribute(newarr, dist=dax)
        return Field(self._tgt(mode), val=newarr)
