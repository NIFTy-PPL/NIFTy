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

import abc

from nifty.operators.linear_operator import LinearOperator


class EndomorphicOperator(LinearOperator):

    # ---Overwritten properties and methods---

    def inverse_times(self, x, spaces=None):
        if self.self_adjoint and self.unitary:
            return self.times(x, spaces)
        else:
            return super(EndomorphicOperator, self).inverse_times(
                                                              x=x,
                                                              spaces=spaces)

    def adjoint_times(self, x, spaces=None):
        if self.self_adjoint:
            return self.times(x, spaces)
        else:
            return super(EndomorphicOperator, self).adjoint_times(
                                                                x=x,
                                                                spaces=spaces)

    def adjoint_inverse_times(self, x, spaces=None):
        if self.self_adjoint:
            return self.inverse_times(x, spaces)
        else:
            return super(EndomorphicOperator, self).adjoint_inverse_times(
                                                                x=x,
                                                                spaces=spaces)

    def inverse_adjoint_times(self, x, spaces=None):
        if self.self_adjoint:
            return self.inverse_times(x, spaces)
        else:
            return super(EndomorphicOperator, self).inverse_adjoint_times(
                                                                x=x,
                                                                spaces=spaces)

    # ---Mandatory properties and methods---

    @property
    def target(self):
        return self.domain

    # ---Added properties and methods---

    @abc.abstractproperty
    def self_adjoint(self):
        raise NotImplementedError
