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

from ..multi import MultiDomain, MultiField
from .linear_operator import LinearOperator


class MultiAdaptor(LinearOperator):
    def __init__(self, target):
        super(MultiAdaptor, self).__init__()
        if not isinstance(target, MultiDomain) or len(target) > 1:
            raise TypeError
        self._target = target
        self._domain = list(target.values())[0]

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        key = list(self.target.keys())[0]
        if mode == self.TIMES:
            return MultiField({key: x})
        else:
            return x[key]
