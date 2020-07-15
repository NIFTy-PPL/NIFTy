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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..linearization import Linearization
from ..minimization.energy import Energy
from ..sugar import makeDomain


class EnergyAdapter(Energy):
    """Helper class which provides the traditional Nifty Energy interface to
    Nifty operators with a scalar target domain.

    Parameters
    -----------
    position: Field or MultiField
        The position where the minimization process is started.
    op: EnergyOperator
        The expression computing the energy from the input data.
    constants: list of strings
        The component names of the operator's input domain which are assumed
        to be constant during the minimization process.
        If the operator's input domain is not a MultiField, this must be empty.
        Default: [].
    want_metric: bool
        If True, the class will provide a `metric` property. This should only
        be enabled if it is required, because it will most likely consume
        additional resources. Default: False.
    nanisinf : bool
        If true, nan energies which can happen due to overflows in the forward
        model are interpreted as inf. Thereby, the code does not crash on
        these occaisions but rather the minimizer is told that the position it
        has tried is not sensible.
    """

    def __init__(self, position, op, constants=[], want_metric=False,
                 nanisinf=False):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        if len(constants) > 0:
            cstpos = position.extract_by_keys(constants)
            _, self._op = op.simplify_for_constant_input(cstpos)
            varkeys = set(op.domain.keys()) - set(constants)
            position = position.extract_by_keys(varkeys)
        self._want_metric = want_metric
        lin = Linearization.make_var(position, want_metric)
        tmp = self._op(lin)
        self._val = tmp.val.val[()]
        self._grad = tmp.gradient
        self._metric = tmp._metric
        self._nanisinf = bool(nanisinf)
        if self._nanisinf and np.isnan(self._val):
            self._val = np.inf

    def at(self, position):
        return EnergyAdapter(position, self._op, want_metric=self._want_metric,
                             nanisinf=self._nanisinf)

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
