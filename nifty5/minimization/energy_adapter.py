from __future__ import absolute_import, division, print_function

from ..compat import *
from ..linearization import Linearization
from ..minimization.energy import Energy
from ..operators.block_diagonal_operator import BlockDiagonalOperator
from ..operators.scaling_operator import ScalingOperator


class EnergyAdapter(Energy):
    def __init__(self, position, op, controller=None, preconditioner=None,
                 constants=[]):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        self._val = self._grad = self._metric = None
        self._controller = controller
        self._preconditioner = preconditioner
        self._constants = constants

    def at(self, position):
        return EnergyAdapter(position, self._op, self._controller,
                             self._preconditioner, self._constants)

    def _fill_all(self):
        if len(self._constants) == 0:
            tmp = self._op(Linearization.make_var(self._position))
        else:
            ops = [ScalingOperator(0. if key in self._constants else 1., dom)
                   for key, dom in self._position.domain.items()]
            bdop = BlockDiagonalOperator(self._position.domain, tuple(ops))
            tmp = self._op(Linearization(self._position, bdop))
        self._val = tmp.val.local_data[()]
        self._grad = tmp.gradient
        if self._controller is not None:
            from ..operators.linear_operator import LinearOperator
            from ..operators.inversion_enabler import InversionEnabler

            if self._preconditioner is None:
                precond = None
            elif isinstance(self._preconditioner, LinearOperator):
                precond = self._preconditioner
            elif isinstance(self._preconditioner, Energy):
                precond = self._preconditioner.at(self._position).metric
            self._metric = InversionEnabler(tmp._metric, self._controller,
                                            precond)
        else:
            self._metric = tmp._metric

    @property
    def value(self):
        if self._val is None:
            self._val = self._op(self._position).local_data[()]
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
