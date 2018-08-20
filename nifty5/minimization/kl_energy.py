from __future__ import absolute_import, division, print_function

from ..compat import *
from .energy import Energy
from ..linearization import Linearization
from ..operators.scaling_operator import ScalingOperator
from ..operators.block_diagonal_operator import BlockDiagonalOperator
from .. import utilities


class KL_Energy(Energy):
    def __init__(self, position, h, nsamp, constants=[], _samples=None):
        super(KL_Energy, self).__init__(position)
        self._h = h
        self._constants = constants
        if _samples is None:
            met = h(Linearization.make_var(position)).metric
            _samples = tuple(met.draw_sample(from_inverse=True)
                             for _ in range(nsamp))
        self._samples = _samples
        if len(constants) == 0:
            tmp = Linearization.make_var(position)
        else:
            ops = [ScalingOperator(0. if key in constants else 1., dom)
                   for key, dom in position.domain.items()]
            bdop = BlockDiagonalOperator(position.domain, tuple(ops))
            tmp = Linearization(position, bdop)
        mymap = map(lambda v: self._h(tmp+v), self._samples)
        tmp = utilities.my_sum(mymap) * (1./len(self._samples))
        self._val = tmp.val.local_data[()]
        self._grad = tmp.gradient
        self._metric = tmp.metric

    def at(self, position):
        return KL_Energy(position, self._h, 0, self._constants, self._samples)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def apply_metric(self, x):
        return self._metric(x)

    @property
    def samples(self):
        return self._samples
