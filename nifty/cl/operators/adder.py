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

from ..field import Field
from ..multi_field import MultiField
from ..sugar import makeField
from .operator import Operator


class Adder(Operator):
    """Adds a fixed field.

    Parameters
    ----------
    a : :class:`nifty.cl.field.Field` or :class:`nifty.cl.multi_field.MultiField` or Scalar
        The field by which the input is shifted.
    """
    def __init__(self, a, neg=False, domain=None):
        if isinstance(a, (Field, MultiField)):
            self._a = a
        elif np.isscalar(a):
            self._a = makeField(domain, a)
        else:
            raise TypeError
        self._domain = self._target = self._a.domain
        self._neg = bool(neg)

    def _device_preparation(self, x):
        self._a = self._a.at(x.device_id)

    def apply(self, x):
        self._check_input(x)
        self._device_preparation(x)
        if self._neg:
            return x - self._a
        return x + self._a
