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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .. import utilities
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..field import Field
from .linear_operator import LinearOperator


class CentralFieldZeroPadder(LinearOperator):
    """Operator which applies central zero-padding to one of the subdomains
    of its input field

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        The operator's input domain.
    new_shape : list or tuple of int
        The new dimensions of the subdomain which is zero-padded.
        No entry must be smaller than the corresponding dimension in the
        operator's domain.
    space : int
        The index of the subdomain to be zero-padded. If None, it is set to 0
        if domain contains exactly one space. domain[space] must be an RGSpace.
    split_even : boolean
        When set to True and padding on an axis with even length, the
        "central" entry will be split up. This is useful for padding in
        harmonic spaces.
    """
    def __init__(self, domain, new_shape, space=0, split_even=False):
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)
        self._split_even = split_even
        dom = self._domain[self._space]
        if not isinstance(dom, RGSpace):
            raise TypeError("RGSpace required")
        if len(new_shape) != len(dom.shape):
            raise ValueError("Shape mismatch")
        if any([a < b for a, b in zip(new_shape, dom.shape)]):
            raise ValueError("New shape must not be smaller than old shape")
        self._target = list(self._domain)
        self._target[self._space] = RGSpace(new_shape, dom.distances,
                                            dom.harmonic)
        self._target = DomainTuple.make(self._target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        v = x.val
        curshp = list(self._dom(mode).shape)
        tgtshp = self._tgt(mode).shape
        for d in self._target.axes[self._space]:
            if v.shape[d] == tgtshp[d]:  # nothing to do
                continue

            idx = (slice(None),) * d

            if mode == self.TIMES:
                shp = list(v.shape)
                shp[d] = tgtshp[d]
                xnew = np.zeros(shp, dtype=v.dtype)
                Nyquist = v.shape[d]//2
                i1 = idx + (slice(0, Nyquist+1),)
                xnew[i1] = v[i1]
                i1 = idx + (slice(None, -(Nyquist+1), -1),)
                xnew[i1] = v[i1]

                if self._split_even and (v.shape[d] & 1) == 0:
                    # even number of pixels
                    i1 = idx+(Nyquist,)
                    xnew[i1] *= 0.5
                    i1 = idx+(-Nyquist,)
                    xnew[i1] *= 0.5

            else:  # ADJOINT_TIMES
                shp = list(v.shape)
                shp[d] = tgtshp[d]
                xnew = np.zeros(shp, dtype=v.dtype)
                Nyquist = xnew.shape[d]//2
                i1 = idx + (slice(0, Nyquist+1),)
                xnew[i1] = v[i1]
                i1 = idx + (slice(None, -(Nyquist+1), -1),)
                xnew[i1] += v[i1]

                if self._split_even and (xnew.shape[d] & 1) == 0:
                    # even number of pixels
                    i1 = idx+(Nyquist,)
                    xnew[i1] *= 0.5

            curshp[d] = xnew.shape[d]
            v = xnew
        return Field(self._tgt(mode), v)


class FieldZeroPadder(LinearOperator):
    """FieldZeroPadder with choosable offset

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        The operator's input domain.
    new_shape : list or tuple of int
        The new dimensions of the subdomain which is zero-padded.
        No entry must be smaller than the corresponding dimension in the
        operator's domain.
    space : int
        The index of the subdomain to be zero-padded. If None, it is set to 0
        if domain contains exactly one space. domain[space] must be an RGSpace.
    offset : tuple of int or None
        Where in the new zero-padded array to place the input field.
        If `None` is given, place the field at zero offset.

    """
    def __init__(self, domain, new_shape, space=0, offset=None):
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)
        dom = self._domain[self._space]
        if offset is None:
            self._offset = (0, ) * len(dom.shape)
        else:
            self._offset = offset
        if not isinstance(dom, RGSpace):
            raise TypeError("RGSpace required")
        if len(new_shape) != len(dom.shape):
            raise ValueError("New shape mismatch")
        if len(self._offset) != len(dom.shape):
            raise ValueError("offset shape mismatch")
        if any([a < b for a, b in zip(new_shape, dom.shape)]):
            raise ValueError("New shape must not be smaller than old shape")
        end_idx = [a + b for a, b in zip(dom.shape, self._offset)]
        if any([a < b for a, b in zip(new_shape, end_idx)]):
            raise ValueError(
                "Input field pasted at offset would overflow target boundaries"
            )
        self._target = list(self._domain)
        self._target[self._space] = RGSpace(new_shape, dom.distances,
                                            dom.harmonic)
        self._target = DomainTuple.make(self._target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        dom = self._domain[self._space]

        num_spaces = len(self._target.axes)
        idx = tuple()
        for i in range(num_spaces):
            for ax_idx in range(len(self._target.axes[i])):
                if i == self._space:
                    ax_offset = self._offset[ax_idx]
                    idx += (slice(ax_offset, dom.shape[ax_idx] + ax_offset), )
                else:
                    idx += (slice(None), )

        if mode == self.TIMES:
            xnew = np.zeros(self._target.shape, dtype=x.val.dtype)
            xnew[idx] = x.val
        else:  # Adjoint times
            xnew = x.val[idx]

        return Field(self._tgt(mode), xnew)
