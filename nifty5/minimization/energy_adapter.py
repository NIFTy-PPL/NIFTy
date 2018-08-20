from __future__ import absolute_import, division, print_function

from ..compat import *
from ..linearization import Linearization
from ..minimization.energy import Energy
from ..operators.block_diagonal_operator import BlockDiagonalOperator
from ..operators.scaling_operator import ScalingOperator


class EnergyAdapter(Energy):
    def __init__(self, position, op, constants=[]):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        self._constants = constants
        if len(self._constants) == 0:
            tmp = self._op(Linearization.make_var(self._position))
        else:
            ops = [ScalingOperator(0. if key in self._constants else 1., dom)
                   for key, dom in self._position.domain.items()]
            bdop = BlockDiagonalOperator(self._position.domain, tuple(ops))
            tmp = self._op(Linearization(self._position, bdop))
        self._val = tmp.val.local_data[()]
        self._grad = tmp.gradient
        self._metric = tmp._metric

    def at(self, position):
        return EnergyAdapter(position, self._op, self._constants)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def apply_metric(self, x):
        return self._metric(x)
