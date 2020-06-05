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
from .block_diagonal_operator import BlockDiagonalOperator
from .energy_operators import EnergyOperator
from .operator import Operator
from .scaling_operator import ScalingOperator
from .simple_linear_operators import NullOperator


class ConstCollector(object):
    def __init__(self):
        self._const = None
        self._nc = set()

    def mult(self, const, fulldom):
        if const is None:
            self._nc |= set(fulldom)
        else:
            self._nc |= set(fulldom) - set(const)
            if self._const is None:
                from ..multi_field import MultiField
                self._const = MultiField.from_dict(
                    {key: const[key] for key in const if key not in self._nc})
            else:
                from ..multi_field import MultiField
                self._const = MultiField.from_dict(
                    {key: self._const[key]*const[key]
                     for key in const if key not in self._nc})

    def add(self, const, fulldom):
        if const is None:
            self._nc |= set(fulldom.keys())
        else:
            from ..multi_field import MultiField
            self._nc |= set(fulldom.keys()) - set(const.keys())
            if self._const is None:
                self._const = MultiField.from_dict(
                    {key: const[key]
                     for key in const.keys() if key not in self._nc})
            else:
                self._const = self._const.unite(const)
                self._const = MultiField.from_dict(
                    {key: self._const[key]
                     for key in self._const if key not in self._nc})

    @property
    def constfield(self):
        return self._const


class ConstantOperator(Operator):
    def __init__(self, dom, output):
        from ..sugar import makeDomain
        self._domain = makeDomain(dom)
        self._target = output.domain
        self._output = output

    def apply(self, x):
        from .simple_linear_operators import NullOperator
        self._check_input(x)
        if x.jac is not None:
            return x.new(self._output, NullOperator(self._domain, self._target))
        return self._output

    def __repr__(self):
        dom = self.domain.keys() if isinstance(self.domain, MultiDomain) else '()'
        tgt = self.target.keys() if isinstance(self.target, MultiDomain) else '()'
        return f'{tgt} <- ConstantOperator <- {dom}'


class SlowPartialConstantOperator(Operator):
    def __init__(self, domain, constant_keys):
        from ..sugar import makeDomain
        if not isinstance(domain, MultiDomain):
            raise TypeError
        if set(constant_keys) > set(domain.keys()) or len(constant_keys) == 0:
            raise ValueError
        self._keys = set(constant_keys) & set(domain.keys())
        self._domain = self._target = makeDomain(domain)

    def apply(self, x):
        self._check_input(x)
        if x.jac is None:
            return x
        jac = {kk: ScalingOperator(dd, 0 if kk in self._keys else 1)
               for kk, dd in self._domain.items()}
        return x.prepend_jac(BlockDiagonalOperator(x.jac.domain, jac))

    def __repr__(self):
        return f'SlowPartialConstantOperator ({self._keys})'


class ConstantEnergyOperator(EnergyOperator):
    def __init__(self, dom, output):
        from ..sugar import makeDomain
        from ..field import Field
        self._domain = makeDomain(dom)
        if not isinstance(output, Field):
            output = Field.scalar(float(output))
        if self.target is not output.domain:
            raise TypeError
        self._output = output

    def apply(self, x):
        self._check_input(x)
        if x.jac is not None:
            val = self._output
            jac = NullOperator(self._domain, self._target)
            met = NullOperator(self._domain, self._domain) if x.want_metric else None
            return x.new(val, jac, met)
        return self._output

    def __repr__(self):
        return 'ConstantEnergyOperator <- {}'.format(self.domain.keys())
