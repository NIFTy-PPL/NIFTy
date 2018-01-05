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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import abc
from ..utilities import NiftyMeta
from ..field import Field
from future.utils import with_metaclass
import numpy as np


class LinearOperator(with_metaclass(
        NiftyMeta, type('NewBase', (object,), {}))):

    _validMode = (False, True, True, False, True, False, False, False, True)
    _inverseMode = (0, 4, 8, 0, 1, 0, 0, 0, 2)
    _inverseCapability = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
    _adjointMode = (0, 2, 1, 0, 8, 0, 0, 0, 4)
    _adjointCapability = (0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15)
    _addInverse = (0, 5, 10, 15, 5, 5, 15, 15, 10, 15, 10, 15, 15, 15, 15, 15)
    _backwards = 6
    TIMES = 1
    ADJOINT_TIMES = 2
    INVERSE_TIMES = 4
    ADJOINT_INVERSE_TIMES = 8
    INVERSE_ADJOINT_TIMES = 8

    def _dom(self, mode):
        return self.domain if (mode & 9) else self.target

    def _tgt(self, mode):
        return self.domain if (mode & 6) else self.target

    def __init__(self):
        pass

    @abc.abstractproperty
    def domain(self):
        """
        domain : DomainTuple
            The domain on which the Operator's input Field lives.
            Every Operator which inherits from the abstract LinearOperator
            base class must have this attribute.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def target(self):
        """
        target : DomainTuple
            The domain on which the Operator's output Field lives.
            Every Operator which inherits from the abstract LinearOperator
            base class must have this attribute.
        """
        raise NotImplementedError

    @property
    def inverse(self):
        from .inverse_operator import InverseOperator
        return InverseOperator(self)

    @property
    def adjoint(self):
        from .adjoint_operator import AdjointOperator
        return AdjointOperator(self)

    @staticmethod
    def _toOperator(thing, dom):
        from .diagonal_operator import DiagonalOperator
        from .scaling_operator import ScalingOperator
        if isinstance(thing, LinearOperator):
            return thing
        if isinstance(thing, Field):
            return DiagonalOperator(thing)
        if np.isscalar(thing):
            return ScalingOperator(thing, dom)
        return NotImplemented

    def __mul__(self, other):
        from .chain_operator import ChainOperator
        other = self._toOperator(other, self.domain)
        return ChainOperator(self, other)

    def __rmul__(self, other):
        from .chain_operator import ChainOperator
        other = self._toOperator(other, self.target)
        return ChainOperator(other, self)

    def __add__(self, other):
        from .sum_operator import SumOperator
        other = self._toOperator(other, self.domain)
        return SumOperator(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from .sum_operator import SumOperator
        other = self._toOperator(other, self.domain)
        return SumOperator(self, other, neg=True)

    # MR FIXME: this might be more complicated ...
    def __rsub__(self, other):
        from .sum_operator import SumOperator
        other = self._toOperator(other, self.domain)
        return SumOperator(other, self, neg=True)

    def supports(self, ops):
        return False

    @abc.abstractproperty
    def capability(self):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, x, mode):
        raise NotImplementedError

    def __call__(self, x):
        return self.apply(x, self.TIMES)

    def times(self, x):
        return self.apply(x, self.TIMES)

    def inverse_times(self, x):
        return self.apply(x, self.INVERSE_TIMES)

    def adjoint_times(self, x):
        return self.apply(x, self.ADJOINT_TIMES)

    def adjoint_inverse_times(self, x):
        return self.apply(x, self.ADJOINT_INVERSE_TIMES)

    def inverse_adjoint_times(self, x):
        return self.apply(x, self.ADJOINT_INVERSE_TIMES)

    def _check_mode(self, mode):
        if not self._validMode[mode]:
            raise ValueError("invalid operator mode specified")
        if mode & self.capability == 0:
            raise ValueError("requested operator mode is not supported")

    def _check_input(self, x, mode):
        if not isinstance(x, Field):
            raise ValueError("supplied object is not a `Field`.")

        self._check_mode(mode)
        if x.domain != self._dom(mode):
                raise ValueError("The operator's and and field's domains "
                                 "don't match.")
