# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function

import itertools

import numpy as np

from .. import dobj, utilities
from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..field import Field
from .linear_operator import LinearOperator


# MR FIXME: for even axis lengths, we probably should split the value at the
#           highest frequency.
class CentralZeroPadder(LinearOperator):
    """Operator that enlarges a fields domain by adding zeros from the middle.

    Parameters
    ---------

    domain: Domain, tuple of Domains or DomainTuple
            The domain of the data that is input by "times" and output by
            "adjoint_times"
    new_shape: tuple
               Shape of the target domain.
    space: int, optional
           The index of the subdomain on which the operator should act
           If None, it is set to 0 if `domain` contains exactly one space.
           `domain[space]` must be an RGSpace.

    """

    def __init__(self, domain, new_shape, space=0):
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)
        dom = self._domain[self._space]

        # verify domains
        if not isinstance(dom, RGSpace):
            raise TypeError("RGSpace required")
        if len(new_shape) != len(dom.shape):
            raise ValueError("Shape mismatch")
        if any([a < b for a, b in zip(new_shape, dom.shape)]):
            raise ValueError("New shape must be larger than old shape")

        # make target space
        tgt = RGSpace(new_shape, dom.distances)
        self._target = list(self._domain)
        self._target[self._space] = tgt
        self._target = DomainTuple.make(self._target)

        self._capability = self.TIMES | self.ADJOINT_TIMES

        # define the axes along which the input field is sliced
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
        v = x.val
        shp_out = self._tgt(mode).shape
        v, x = dobj.ensure_not_distributed(v, self._target.axes[self._space])
        curax = dobj.distaxis(v)

        if mode == self.TIMES:
            # slice along each axis and copy the data to an
            # array of zeros which has the shape of the target domain
            y = np.zeros(dobj.local_shape(shp_out, curax), dtype=x.dtype)
            for i in self.slicer:
                y[i] = x[i]
        else:
            # slice along each axis and copy the data to an array of zeros
            # which has the shape of the input domain to remove excess zeros
            y = np.empty(dobj.local_shape(shp_out, curax), dtype=x.dtype)
            for i in self.slicer:
                y[i] = x[i]
        v = dobj.from_local_data(shp_out, y, distaxis=dobj.distaxis(v))
        return Field(self._tgt(mode), dobj.ensure_default_distributed(v))
