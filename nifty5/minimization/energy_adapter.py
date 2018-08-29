from __future__ import absolute_import, division, print_function

from ..compat import *
from ..linearization import Linearization
from ..minimization.energy import Energy
from ..operators.block_diagonal_operator import BlockDiagonalOperator
from ..operators.scaling_operator import ScalingOperator


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
