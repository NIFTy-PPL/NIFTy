from __future__ import absolute_import, division, print_function

import numpy as np
import itertools

from ..compat import *
from .. import utilities
from .linear_operator import LinearOperator
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..field import Field
from .. import dobj


# MR FIXME: for even axis lengths, we probably should split the value at the
#           highest frequency.
class CentralZeroPadder(LinearOperator):
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

        tgt = RGSpace(new_shape, dom.distances)
        self._target = list(self._domain)
        self._target[self._space] = tgt
        self._target = DomainTuple.make(self._target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        slicer = []
        axes = self._target.axes[self._space]
        for i in range(len(self._domain.shape)):
            if i in axes:
                slicer_fw = slice(0, (self._domain.shape[i]+1)//2)
                slicer_bw = slice(-1, -1-(self._domain.shape[i]//2), -1)
                slicer.append((slicer_fw, slicer_bw))
        self.slicer = list(itertools.product(*slicer))

        for i in range(len(self.slicer)):
            for j in range(len(self._domain.shape)):
                if j not in axes:
                    tmp = list(self.slicer[i])
                    tmp.insert(j, slice(None))
                    self.slicer[i] = tuple(tmp)
        self.slicer = tuple(self.slicer)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        dax = dobj.distaxis(x)
        shp_in = x.shape
        shp_out = self._tgt(mode).shape
        axes = self._target.axes[self._space]
        if dax in axes:
            x = dobj.redistribute(x, nodist=axes)
        curax = dobj.distaxis(x)
        x = dobj.local_data(x)

        if mode == self.TIMES:
            y = np.zeros(dobj.local_shape(shp_out, curax), dtype=x.dtype)
            for i in self.slicer:
                y[i] = x[i]
        else:
            y = np.empty(dobj.local_shape(shp_out, curax), dtype=x.dtype)
            for i in self.slicer:
                y[i] = x[i]
        y = dobj.from_local_data(shp_out, y, distaxis=curax)
        if dax in axes:
            y = dobj.redistribute(y, dist=dax)
        return Field(self._tgt(mode), val=y)
