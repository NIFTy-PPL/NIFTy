from __future__ import absolute_import, division, print_function

from ..compat import *
from ..minimization.energy import Energy
from ..linearization import Linearization
import numpy as np


class EnergyAdapter(Energy):
    def __init__(self, position, op):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        self._val = self._grad = self._metric = None

    def at(self, position):
        return EnergyAdapter(position, self._op)

    def _fill_all(self):
        tmp = self._op(Linearization.make_var(self._position))
        self._val = tmp.val
        if not np.isscalar(self._val):
            self._val = self._val.local_data[()]
        self._grad = tmp.gradient
        self._metric = tmp.metric

    @property
    def value(self):
        if self._val is None:
            self._val = self._op(self._position)
            if not np.isscalar(self._val):
                self._val = self._val.local_data[()]
        return self._val

    @property
    def gradient(self):
        if self._grad is None:
            self._fill_all()
        return self._grad

    @property
    def metric(self):
        if self._metric is None:
            self._fill_all()
        return self._metric
