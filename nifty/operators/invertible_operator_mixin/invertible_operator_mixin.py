# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

from nifty.minimization import ConjugateGradient

from nifty.field import Field


class InvertibleOperatorMixin(object):
    def __init__(self, inverter=None, preconditioner=None, *args, **kwargs):
        self.__preconditioner = preconditioner
        if inverter is not None:
            self.__inverter = inverter
        else:
            self.__inverter = ConjugateGradient(
                                        preconditioner=self.__preconditioner)
        super(InvertibleOperatorMixin, self).__init__(*args, **kwargs)

    def _times(self, x, spaces, x0=None):
        if x0 is None:
            x0 = Field(self.target, val=0., dtype=x.dtype)

        (result, convergence) = self.__inverter(A=self.inverse_times,
                                                b=x,
                                                x0=x0)
        return result

    def _adjoint_times(self, x, spaces, x0=None):
        if x0 is None:
            x0 = Field(self.domain, val=0., dtype=x.dtype)

        (result, convergence) = self.__inverter(A=self.adjoint_inverse_times,
                                                b=x,
                                                x0=x0)
        return result

    def _inverse_times(self, x, spaces, x0=None):
        if x0 is None:
            x0 = Field(self.domain, val=0., dtype=x.dtype)

        (result, convergence) = self.__inverter(A=self.times,
                                                b=x,
                                                x0=x0)
        return result

    def _adjoint_inverse_times(self, x, spaces, x0=None):
        if x0 is None:
            x0 = Field(self.target, val=0., dtype=x.dtype)

        (result, convergence) = self.__inverter(A=self.adjoint_times,
                                                b=x,
                                                x0=x0)
        return result

    def _inverse_adjoint_times(self, x, spaces):
        raise NotImplementedError(
            "no generic instance method 'inverse_adjoint_times'.")
