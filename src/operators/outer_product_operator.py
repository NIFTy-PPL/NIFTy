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
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import partial

import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from ..multi_field import MultiField
from .linear_operator import LinearOperator


class OuterProduct(LinearOperator):
    """Performs the point-wise outer product of two fields.

    Parameters
    ---------
    domain : DomainTuple, the domain of the input field
    field : :class:`nifty8.field.Field`
    ---------
    """
    def __init__(self, domain, field):
        self._domain = DomainTuple.make(domain)
        self._field = field
        self._target = DomainTuple.make(
            tuple(sub_d for sub_d in field.domain._dom + self._domain._dom))
        self._capability = self.TIMES | self.ADJOINT_TIMES

        try:
            from jax import numpy as jnp
            from jax.tree_util import tree_map

            from ..re import Vector as ReField

            a_j = ReField(field.val) if isinstance(field, (Field, MultiField)) else field

            def jax_expr(x):
                # Preserve the input type
                if not isinstance(x, ReField):
                    a_astype_x = a_j.tree if isinstance(a_j, ReField) else a_j
                else:
                    a_astype_x = a_j

                return tree_map(
                    partial(jnp.tensordot, axes=((), ())),
                    a_astype_x, x
                )

            self._jax_expr = jax_expr
        except ImportError:
            self._jax_expr = None

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return Field(
                self._target, np.multiply.outer(
                    self._field.val, x.val))
        axes = len(self._field.shape)
        return Field(
            self._domain, np.tensordot(self._field.val, x.val, axes))
