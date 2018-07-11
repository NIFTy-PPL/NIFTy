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

from __future__ import absolute_import, division, print_function

from ..compat import *
from ..nonlinearities import Exponential, PositiveTanh, Tanh
from ..sugar import makeOp
from .model import Model


class LocalModel(Model):
    def __init__(self, inp, nonlinearity):
        """
        Computes nonlinearity(inp)
            - LocalModel.value = nonlinearity(value) (pointwise)
            - LocalModel.jacobian = Outer Product of Jacobians

        Parameters
        ----------
        inp : Model
            The model for which the  nonlinarity will be applied.
        nonlinearity: Function
            The nonlinearity to be applied to model.
        """
        super(LocalModel, self).__init__(inp.position)
        self._inp = inp
        self._nonlinearity = nonlinearity
        self._value = nonlinearity(self._inp.value)
        d_inner = self._inp.jacobian
        d_outer = makeOp(self._nonlinearity.derivative(self._inp.value))
        self._jacobian = d_outer * d_inner

    def at(self, position):
        return self.__class__(self._inp.at(position), self._nonlinearity)


def PointwiseExponential(inp):
    return LocalModel(inp, Exponential())


def PointwiseTanh(inp):
    return LocalModel(inp, Tanh())


def PointwisePositiveTanh(inp):
    return LocalModel(inp, PositiveTanh())
