from __future__ import absolute_import, division, print_function

from ..compat import *
from ..minimization.energy import Energy
from ..linearization import Linearization


class EnergyAdapter(Energy):
    def __init__(self, position, op):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        pvar = Linearization.make_var(position)
        self._res = op(pvar)

    def at(self, position):
        return EnergyAdapter(position, self._op)

    @property
    def value(self):
        return self._res.val.local_data[()]

    @property
    def gradient(self):
        return self._res.gradient

    @property
    def metric(self):
        return self._res.metric
