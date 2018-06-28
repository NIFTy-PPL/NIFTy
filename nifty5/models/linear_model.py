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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from ..operators.selection_operator import SelectionOperator
from .model import Model


class LinearModel(Model):
    def __init__(self, inp, lin_op):
        """Computes lin_op(inp) where lin_op is a Linear Operator

        Parameters
        ----------
        inp : Model

        lin_op : LinearOperator
            linear function to be applied to model

        Returns
        -------
        Model
            Model with linear Operator applied:
                - Model.value = LinOp (inp.value) [key-wise]
                - Gradient = LinOp * inp.gradient

        """
        from ..operators.linear_operator import LinearOperator
        super(LinearModel, self).__init__(inp.position)

        if not isinstance(lin_op, LinearOperator):
            raise TypeError("needs a LinearOperator as input")

        self._lin_op = lin_op
        self._inp = inp
        if isinstance(self._lin_op, SelectionOperator):
            self._lin_op = SelectionOperator(self._inp.value.domain,
                                             self._lin_op._key)

        self._value = self._lin_op(self._inp.value)
        self._gradient = self._lin_op*self._inp.gradient

    def at(self, position):
        return self.__class__(self._inp.at(position), self._lin_op)
