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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..field import Field
from ..multi_field import MultiField
from .operator import Operator
from .diagonal_operator import DiagonalOperator
from ..linearization import Linearization
from ..sugar import from_local_data
from numpy import log1p


class Log1p(Operator):
    """computes x -> log(1+x)
    """
    def __init__(self, dom):
        self._domain = dom
        self._target = dom

    def apply(self, x):
        lin = isinstance(x, Linearization)
        xval = x.val if lin else x
        xlval = xval.local_data
        res = from_local_data(xval.domain, log1p(xlval))
        if not lin:
            return res
        jac = DiagonalOperator(1/(1+xval))
        return x.new(res, jac@x.jac)
