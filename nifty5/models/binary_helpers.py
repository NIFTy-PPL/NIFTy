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

from ..multi.multi_field import MultiField
from ..sugar import makeOp
from .model import Model


def _joint_position(op1, op2):
    a = op1.position._val
    b = op2.position._val
    # Note: In python >3.5 one could do {**a, **b}
    ab = a.copy()
    ab.update(b)
    return MultiField(ab)


class ScalarMul(Model):
    """Class representing a model multiplied by a scalar factor."""
    def __init__(self, factor, op):
        # TODO op -> model
        super(ScalarMul, self).__init__(op.position)
        # TODO -> floating
        if not isinstance(factor, (float, int)):
            raise TypeError

        self._op = op
        self._factor = factor

        self._value = self._factor * self._op.value
        self._gradient = self._factor * self._op.gradient

    def at(self, position):
        return self.__class__(self._factor, self._op.at(position))


class Add(Model):
    """Class representing the sum of two models."""
    def __init__(self, position, op1, op2):
        super(Add, self).__init__(position)

        self._op1 = op1.at(position)
        self._op2 = op2.at(position)

        self._value = self._op1.value + self._op2.value
        self._gradient = self._op1.gradient + self._op2.gradient

    @staticmethod
    def make(op1, op2):
        position = _joint_position(op1, op2)
        return Add(position, op1, op2)

    def at(self, position):
        return self.__class__(position, self._op1, self._op2)


class Mul(Model):
    """Class representing the pointwise product of two models."""
    def __init__(self, position, op1, op2):
        super(Mul, self).__init__(position)

        self._op1 = op1.at(position)
        self._op2 = op2.at(position)

        self._value = self._op1.value * self._op2.value
        self._gradient = (makeOp(self._op1.value) * self._op2.gradient +
                          makeOp(self._op2.value) * self._op1.gradient)

    @staticmethod
    def make(op1, op2):
        position = _joint_position(op1, op2)
        return Mul(position, op1, op2)

    def at(self, position):
        return self.__class__(position, self._op1, self._op2)
