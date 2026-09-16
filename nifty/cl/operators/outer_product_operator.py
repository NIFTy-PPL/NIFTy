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
# Copyright(C) 2013-2020 Max-Planck-Society
# Copyright(C) 2026 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from ..utilities import iscomplextype
from .linear_operator import LinearOperator


class OuterProduct(LinearOperator):
    """Performs the point-wise outer product of two fields.

    Parameters
    ---------
    domain : DomainTuple, the domain of the input field
    field : :class:`nifty.cl.field.Field`
    flip : bool
        If False, `field` becomes the leading and the input field the trailing
        factor of the outer product, i.e. `x -> field * x`. If True, the
        order is reversed, i.e. `x -> x * field`. Default: False.
    ---------
    """
    def __init__(self, domain, field, flip=False):
        self._domain = DomainTuple.make(domain)
        self._field = field
        self._flip = bool(flip)
        self._complex = iscomplextype(field.dtype)
        if self._flip:
            doms = tuple(self._domain) + tuple(field.domain)
        else:
            doms = tuple(field.domain) + tuple(self._domain)
        self._target = DomainTuple.make(doms)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def _device_preparation(self, x, mode):
        self._field = self._field.at(x.device_id)

    def apply(self, x, mode):
        self._check_input(x, mode)
        self._device_preparation(x, mode)

        fval = self._field.val
        if mode == self.ADJOINT_TIMES and self._complex:
            fval = np.conj(fval)
        args = [fval, x.val]
        if self._flip:
            args = args[::-1]
        if mode == self.TIMES:
            res = np.multiply.outer(*args)
        else:
            axes = len(self._field.shape)
            res = np.tensordot(*args, axes=axes)
        return Field(self._tgt(mode), res)
