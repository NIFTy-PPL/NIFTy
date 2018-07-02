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


def _joint_position(model1, model2):
    a = model1.position._val
    b = model2.position._val
    # Note: In python >3.5 one could do {**a, **b}
    ab = a.copy()
    ab.update(b)
    return MultiField(ab)


class ScalarMul(Model):
    """Class representing a model multiplied by a scalar factor."""
    def __init__(self, factor, model):
        super(ScalarMul, self).__init__(model.position)
        # TODO -> floating
        if not isinstance(factor, (float, int)):
            raise TypeError

        self._model = model
        self._factor = factor

        self._value = self._factor * self._model.value
        self._gradient = self._factor * self._model.gradient

    def at(self, position):
        return self.__class__(self._factor, self._model.at(position))


class Add(Model):
    """Class representing the sum of two models."""
    def __init__(self, position, model1, model2):
        super(Add, self).__init__(position)

        self._model1 = model1.at(position)
        self._model2 = model2.at(position)

        self._value = self._model1.value + self._model2.value
        self._gradient = self._model1.gradient + self._model2.gradient

    @staticmethod
    def make(model1, model2):
        """Build the sum of two models.

        Parameters
        ----------
        model1: Model
            First model.
        model2: Model
            Second model
        """

        position = _joint_position(model1, model2)
        return Add(position, model1, model2)

    def at(self, position):
        return self.__class__(position, self._model1, self._model2)


class Mul(Model):
    """Class representing the pointwise product of two models."""
    def __init__(self, position, model1, model2):
        super(Mul, self).__init__(position)

        self._model1 = model1.at(position)
        self._model2 = model2.at(position)

        self._value = self._model1.value * self._model2.value
        self._gradient = (makeOp(self._model1.value) * self._model2.gradient +
                          makeOp(self._model2.value) * self._model1.gradient)

    @staticmethod
    def make(model1, model2):
        """Build the pointwise product of two models.

        Parameters
        ----------
        model1: Model
            First model.
        model2: Model
            Second model
        """
        position = _joint_position(model1, model2)
        return Mul(position, model1, model2)

    def at(self, position):
        return self.__class__(position, self._model1, self._model2)
