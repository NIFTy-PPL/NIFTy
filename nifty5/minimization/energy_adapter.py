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

from ..linearization import Linearization
from ..minimization.energy import Energy


class EnergyAdapter(Energy):
    def __init__(self, position, op, constants=[], want_metric=False):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        self._constants = constants
        self._want_metric = want_metric
        lin = Linearization.make_partial_var(position, constants, want_metric)
        tmp = self._op(lin)
        self._val = tmp.val.local_data[()]
        self._grad = tmp.gradient
        self._metric = tmp._metric

    def at(self, position):
        return EnergyAdapter(position, self._op, self._constants,
                             self._want_metric)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    @property
    def metric(self):
        return self._metric

    def apply_metric(self, x):
        return self._metric(x)
