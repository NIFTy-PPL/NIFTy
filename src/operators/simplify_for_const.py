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
# Copyright(C) 2013-2022 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..multi_domain import MultiDomain
from ..utilities import myassert
from .block_diagonal_operator import BlockDiagonalOperator
from .energy_operators import EnergyOperator, LikelihoodEnergyOperator
from .operator import Operator
from .scaling_operator import ScalingOperator
from .simple_linear_operators import NullOperator


class ConstCollector:
    def __init__(self):
        self._const = None  # MultiField on the part of the MultiDomain that could be constant
        self._nc = set()  # NoConstant - set of keys that we know cannot be constant

    def mult(self, const, fulldom):
        if const is None:
            self._nc |= set(fulldom.keys())
        else:
            from ..multi_field import MultiField
            self._nc |= set(fulldom.keys()) - set(const.keys())
            if self._const is None:
                self._const = MultiField.from_dict(
                    {key: const[key]
                     for key in const.keys() if key not in self._nc})
            else:  # we know that the domains are identical for products
                self._const = MultiField.from_dict(
                    {key: self._const[key]*const[key]
                     for key in const.keys() if key not in self._nc})

    def add(self, const, fulldom):
        if const is None:
            self._nc |= set(fulldom.keys())
        else:
            from ..multi_field import MultiField
            self._nc |= set(fulldom.keys()) - set(const.keys())
            self._const = const if self._const is None else self._const.unite(const)
            self._const = MultiField.from_dict(
                {key: const[key]
                 for key in const.keys() if key not in self._nc})

    @property
    def constfield(self):
        return self._const


class ConstantOperator(Operator):
    def __init__(self, output, domain={}):
        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._target = makeDomain(output.domain)
        self._output = output

    def apply(self, x):
        from .simple_linear_operators import NullOperator
        self._check_input(x)
        if x.jac is not None:
            return x.new(self._output, NullOperator(self._domain, self._target))
        return self._output

    def __repr__(self):
        tgt = self.target.keys() if isinstance(self.target, MultiDomain) else '()'
        return f'{tgt} <- ConstantOperator'


class ConstantEnergyOperator(EnergyOperator):
    def __init__(self, output):
        from ..field import Field
        from ..sugar import makeDomain
        self._domain = makeDomain({})
        if not isinstance(output, Field):
            output = Field.scalar(float(output))
        self._output = output

    def apply(self, x):
        self._check_input(x)
        if x.jac is not None:
            val = self._output
            jac = NullOperator(self._domain, self._target)
            met = NullOperator(self._domain, self._domain) if x.want_metric else None
            return x.new(val, jac, met)
        return self._output


class ConstantLikelihoodEnergyOperator(LikelihoodEnergyOperator):
    def __init__(self, output):
        super(ConstantLikelihoodEnergyOperator, self).__init__(None, None)
        op = ConstantEnergyOperator(output)
        self.apply = op.apply
        self._domain = op._domain


class InsertionOperator(Operator):
    def __init__(self, target, cst_field):
        from ..multi_field import MultiField
        from ..sugar import makeDomain
        if not isinstance(target, MultiDomain):
            raise TypeError
        if not isinstance(cst_field, MultiField):
            raise TypeError
        self._target = MultiDomain.make(target)
        cstdom = cst_field.domain
        vardom = makeDomain({kk: vv for kk, vv in self._target.items()
                             if kk not in cst_field.keys()})
        self._domain = vardom
        self._cst = cst_field
        jac = {kk: ScalingOperator(vv, 1.) for kk, vv in self._domain.items()}
        self._jac = BlockDiagonalOperator(self._domain, jac) + NullOperator(makeDomain({}), cstdom)

    def apply(self, x):
        myassert(len(set(self._cst.keys()) & set(x.domain.keys())) == 0)
        val = x if x.jac is None else x.val
        val = val.unite(self._cst)
        if x.jac is None:
            return val
        return x.new(val, self._jac)

    def __repr__(self):
        from ..utilities import indent
        subs = f'Constant: {self._cst.keys()}\nVariable: {self._domain.keys()}'
        return 'InsertionOperator\n'+indent(subs)
