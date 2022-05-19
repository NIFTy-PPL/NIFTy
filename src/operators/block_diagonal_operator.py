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

from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..utilities import check_dtype_or_none, check_object_identity, indent
from .endomorphic_operator import EndomorphicOperator
from .linear_operator import LinearOperator


class BlockDiagonalOperator(EndomorphicOperator):
    """
    Parameters
    ----------
    domain : MultiDomain
        Domain and target of the operator.
    operators : dict
        Dictionary with subdomain names as keys and :class:`LinearOperator` s
        as items. Any missing item will be treated as unity operator.
    """
    def __init__(self, domain, operators):
        if not isinstance(domain, MultiDomain):
            raise TypeError("MultiDomain expected")
        self._domain = domain
        self._ops = tuple(operators[key] if key in operators else None for key in domain.keys())
        self._capability = self._all_ops

        self._dtype = {kk: oo.sampling_dtype for kk, oo in operators.items()}
        if all(vv is None for vv in self._dtype.values()):
            self._dtype = None
        check_dtype_or_none(self._dtype, self._domain)

        for op in self._ops:
            if op is not None:
                if isinstance(op, LinearOperator):
                    if op.target is not op.domain:
                        raise TypeError("domain and target mismatch")
                    self._capability &= op.capability
                else:
                    raise TypeError("LinearOperator expected")

    def get_sqrt(self):
        ops = {}
        for ii, kk in enumerate(self._domain.keys()):
            if self._ops[ii] is None:
                continue
            ops[kk] = self._ops[ii].get_sqrt()
        return BlockDiagonalOperator(self._domain, ops)

    def apply(self, x, mode):
        self._check_input(x, mode)
        val = tuple(op.apply(v, mode=mode) if op is not None else v
                    for op, v in zip(self._ops, x.values()))
        return MultiField(self._domain, val)

    def draw_sample(self, from_inverse=False):
        from ..sugar import from_random
        val = []
        for op, key in zip(self._ops, self._domain.keys()):
            if op is None:
                if self._dtype is None or key not in self._dtype:
                    raise RuntimeError("Need to specify dtype for all operators "
                                       f"that are set to None (key: {key}).")
                a = from_random(self._domain[key], 'normal', dtype=self.sampling_dtype[key])
            else:
                a = op.draw_sample(from_inverse)
            val.append(a)
        return MultiField(self._domain, tuple(val))

    def _combine_chain(self, op):
        check_object_identity(self._domain, op._domain)
        res = {key: v1(v2)
               for key, v1, v2 in zip(self._domain.keys(), self._ops, op._ops)}
        return BlockDiagonalOperator(self._domain, res)

    def _combine_sum(self, op, selfneg, opneg):
        from ..operators.sum_operator import SumOperator
        check_object_identity(self._domain, op._domain)
        res = {key: SumOperator.make([v1, v2], [selfneg, opneg])
               for key, v1, v2 in zip(self._domain.keys(), self._ops, op._ops)}
        return BlockDiagonalOperator(self._domain, res)

    def __repr__(self):
        s = []
        for ii, kk in enumerate(self.domain.keys()):
            if self._ops[ii] is None:
                ss = f"{kk}: id"
                if self._dtype is not None and kk in self._dtype:
                    ss += f" (sampling dtype: {self._dtype[kk]})"
                s.append(ss)
            else:
                s.append(f'{kk}: {self._ops[ii]}')
        s = "\n".join(s)
        return 'BlockDiagonalOperator:\n' + indent(s)
