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


class FieldZeroPadder(LinearOperator):
    """Operator which applies zero-padding to one of the subdomains of its
    input field.

    If no indication is given as to where the operator should pad, the
    padding is added to the end of the subdomain chosen by `space`.

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
    padding_position : string, optional
        Where to add the padding. Valid values are 'start', 'center' and 'end'.
    central : bool, optional, DEPRECATED
        `True` is equivalent to ``padding_position = 'middle'``.

    Notes
    -----
    When doing central padding on an axis with an even length, the "central"
    entry should in principle be split up; this is currently not done.
    """
    def __init__(self, domain, new_shape, space=0, padding_position=None,
                 central=None):
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)
        self._position = 'end'
        if padding_position is not None:
            if central is not None: raise ValueError()
            if padding_position not in ['start', 'center', 'end']:
                raise ValueError("Invalid 'padding_position' value given")
            self._position = padding_position
        elif central:
            self._position = 'center'
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
                if self._position == 'center':
                    Nyquist = v.shape[d]//2
                    i1 = idx + (slice(0, Nyquist+1),)
                    xnew[i1] = v[i1]
                    i1 = idx + (slice(None, -(Nyquist+1), -1),)
                    xnew[i1] = v[i1]
#                     if (v.shape[d] & 1) == 0:  # even number of pixels
#                         i1 = idx+(Nyquist,)
#                         xnew[i1] *= 0.5
#                         i1 = idx+(-Nyquist,)
#                         xnew[i1] *= 0.5
                elif self._position == 'end':
                    xnew[idx + (slice(0, v.shape[d]),)] = v
                else:
                    d_len = tgtshp[d]
                    xnew[idx + (slice(d_len - v.shape[d], d_len),)] = v
            else:  # ADJOINT_TIMES
                if self._position == 'center':
                    shp = list(v.shape)
                    shp[d] = tgtshp[d]
                    xnew = np.zeros(shp, dtype=v.dtype)
                    Nyquist = xnew.shape[d]//2
                    i1 = idx + (slice(0, Nyquist+1),)
                    xnew[i1] = v[i1]
                    i1 = idx + (slice(None, -(Nyquist+1), -1),)
                    xnew[i1] += v[i1]
#                     if (xnew.shape[d] & 1) == 0:  # even number of pixels
#                         i1 = idx+(Nyquist,)
#                         xnew[i1] *= 0.5
                elif self._position == 'end':
                    xnew = v[idx + (slice(0, tgtshp[d]),)]
                else:
                    d_len = v.shape[d]
                    xnew = v[idx + (slice(d_len - tgtshp[d], d_len),)]

            curshp[d] = xnew.shape[d]
            v = xnew
        return Field(self._tgt(mode), v)
