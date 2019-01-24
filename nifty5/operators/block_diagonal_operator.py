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

from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from .endomorphic_operator import EndomorphicOperator


class BlockDiagonalOperator(EndomorphicOperator):
    """
    Parameters
    ----------
    domain : MultiDomain
        Domain and target of the operator.
    operators : dict
        Dictionary with subdomain names as keys and :class:`LinearOperator` s
        as items.
    """
    def __init__(self, domain, operators):
        if not isinstance(domain, MultiDomain):
            raise TypeError("MultiDomain expected")
        self._domain = domain
        self._ops = tuple(operators[key] for key in domain.keys())
        self._capability = self._all_ops
        for op in self._ops:
            if op is not None:
                self._capability &= op.capability

    def apply(self, x, mode):
        self._check_input(x, mode)
        val = tuple(op.apply(v, mode=mode) if op is not None else None
                    for op, v in zip(self._ops, x.values()))
        return MultiField(self._domain, val)

#    def draw_sample(self, from_inverse=False, dtype=np.float64):
#        dtype = MultiField.build_dtype(dtype, self._domain)
#        val = tuple(op.draw_sample(from_inverse, dtype)
#                    for op in self._op)
#        return MultiField(self._domain, val)

    def _combine_chain(self, op):
        if self._domain != op._domain:
            raise ValueError("domain mismatch")
        res = {key: v1(v2)
               for key, v1, v2 in zip(self._domain.keys(), self._ops, op._ops)}
        return BlockDiagonalOperator(self._domain, res)

    def _combine_sum(self, op, selfneg, opneg):
        from ..operators.sum_operator import SumOperator
        if self._domain != op._domain:
            raise ValueError("domain mismatch")
        res = {key: SumOperator.make([v1, v2], [selfneg, opneg])
               for key, v1, v2 in zip(self._domain.keys(), self._ops, op._ops)}
        return BlockDiagonalOperator(self._domain, res)
