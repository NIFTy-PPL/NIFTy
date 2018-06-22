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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from ..operators.selection_operator import SelectionOperator
from ..operators.diagonal_operator import DiagonalOperator
from ..utilities import NiftyMetaBase
from ..field import Field


class Model(NiftyMetaBase()):
    def __init__(self, position):
        self._position = position

    def at(self, position):
        raise NotImplementedError

    @property
    def position(self):
        return self._position

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    def __getitem__(self, key):
        sel = SelectionOperator(self.value.domain, key)
        return sel(self)

    def __add__(self, other):
        if not isinstance(other, Model):
            raise TypeError
        from .binary_helpers import Add
        return Add.make(self, other)

    def __sub__(self, other):
        return self.__add__(self, (-1) * other)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            from .binary_helpers import ScalarMul
            return ScalarMul(other, self)
        if isinstance(other, Model):
            from .binary_helpers import Mul
            return Mul.make(self, other)
        if isinstance(other, Field):
            return DiagonalOperator(other)(self)
        raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, (float, int, Field)):
            return self.__mul__(other)
        raise NotImplementedError
