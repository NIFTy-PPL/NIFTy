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

from operator import add, sub

import numpy as np

from ..field import Field
from ..multi_field import MultiField
from ..sugar import makeDomain
from .operator import Operator


class Adder(Operator):
    """Adds a fixed field.

    Parameters
    ----------
    a : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField` or Scalar
        The field by which the input is shifted.
    """
    def __init__(self, a, neg=False, domain=None):
        self._a = a
        if isinstance(a, (Field, MultiField)):
            dom = a.domain
        elif np.isscalar(a):
            dom = makeDomain(domain)
        else:
            raise TypeError
        self._domain = self._target = dom
        self._neg = bool(neg)

        try:
            from jax.tree_util import tree_map

            from ..re import Vector as ReField

            a_j = ReField(a.val) if isinstance(a, (Field, MultiField)) else a

            def jax_expr(x):
                # Preserve the input type
                if not isinstance(x, ReField):
                    a_astype_x = a_j.tree if isinstance(a_j, ReField) else a_j
                else:
                    a_astype_x = a_j
                return tree_map(sub if neg else add, x, a_astype_x)

            self._jax_expr = jax_expr
        except ImportError:
            self._jax_expr = None

    def apply(self, x):
        self._check_input(x)
        if self._neg:
            return x - self._a
        return x + self._a
