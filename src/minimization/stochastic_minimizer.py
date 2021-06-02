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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .minimizer import Minimizer
from .energy import Energy
from .kl_energies import _SelfAdjointOperatorWrapper, _get_lo_hi
from ..linearization import Linearization
from ..utilities import myassert, allreduce_sum
from ..multi_domain import MultiDomain
from ..sugar import from_random
from .. import random


class _StochasticEnergyAdapter(Energy):
    def __init__(self, position, local_ops, n_samples, comm, nanisinf):
        super(_StochasticEnergyAdapter, self).__init__(position)
        for op in local_ops:
            myassert(position.domain == op.domain)
        self._comm = comm
        self._local_ops = local_ops
        self._n_samples = n_samples
        lin = Linearization.make_var(position)
        v, g = [], []
        for op in self._local_ops:
            tmp = op(lin)
            v.append(tmp.val.val)
            g.append(tmp.gradient)
        self._val = allreduce_sum(v, self._comm)[()]/self._n_samples
        if np.isnan(self._val) and self._nanisinf:
            self._val = np.inf
        self._grad = allreduce_sum(g, self._comm)/self._n_samples

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def at(self, position):
        return _StochasticEnergyAdapter(position, self._local_ops,
                        self._n_samples, self._comm, self._nanisinf)

    def apply_metric(self, x):
        lin = Linearization.make_var(self.position, want_metric=True)
        res = []
        for op in self._local_ops:
            res.append(op(lin).metric(x))
        return allreduce_sum(res, self._comm)/self._n_samples

    @property
    def metric(self):
        return _SelfAdjointOperatorWrapper(self.position.domain,
                                           self.apply_metric)


class PartialSampledEnergy(_StochasticEnergyAdapter):
    def __init__(self, position, op, keys, local_ops, n_samples, comm, nanisinf,
                 _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(PartialSampledEnergy, self).__init__(position,
             local_ops, n_samples, comm, nanisinf)
        self._op = op
        self._keys = keys

    def at(self, position):
        return PartialSampledEnergy(position, self._op, self._keys,
                                    self._local_ops, self._n_samples,
                                    self._comm, self._nanisinf,
                                    _callingfrommake=True)

    def resample_at(self, position):
        return PartialSampledEnergy.make(position, self._op, self._keys,
                                         self._n_samples, self._comm)
    
    @staticmethod
    def make(position, op, keys, n_samples, mirror_samples, nanisinf = False, comm=None):
        samdom = {}
        for k in keys:
            if k in position.domain.keys():
                raise ValueError
            if k not in op.domain.keys():
                raise ValueError
            else:
                samdom[k] = op.domain[k]
        samdom = MultiDomain.make(samdom)
        local_ops = []
        sseq = random.spawn_sseq(n_samples)
        for i in range(*_get_lo_hi(comm, n_samples)):
            with random.Context(sseq[i]):
                rnd = from_random(samdom)
                _, tmp = op.simplify_for_constant_input(rnd)
                myassert(tmp.domain == position.domain)
                local_ops.append(tmp)
                if mirror_samples:
                    local_ops.append(op.simplify_for_constant_input(-rnd)[1])
        n_samples = 2*n_samples if mirror_samples else n_samples
        return PartialSampledEnergy(position, op, keys, local_ops, n_samples,
                                    comm, nanisinf, _callingfrommake=True)


class ADVIOptimizer(Minimizer):
    """Provide an implementation of an adaptive step-size sequence optimizer,
    following https://arxiv.org/abs/1603.00788.

    Parameters
    ----------
    steps: int
        The number of concecutive steps during one call of the optimizer.
    eta: positive float
        The scale of the step-size sequence. It might have to be adapted to the
        application to increase performance. Default: 1.
    alpha: float between 0 and 1
        The fraction of how much the current gradient impacts the momentum.
    tau: positive float
        This quantity prevents division by zero.
    epsilon: positive float
        A small value guarantees Robbins and Monro conditions.
    """

    def __init__(self, steps, eta=1, alpha=0.1, tau=1, epsilon=1e-16):
        self.alpha = alpha
        self.eta = eta
        self.tau = tau
        self.epsilon = epsilon
        self.counter = 1
        self.steps = steps
        self.s = None

    def _step(self, position, gradient):
        self.s = self.alpha * gradient ** 2 + (1 - self.alpha) * self.s
        self.rho = (
            self.eta
            * self.counter ** (-0.5 + self.epsilon)
            / (self.tau + (self.s).sqrt())
        )
        new_position = position - self.rho * gradient
        self.counter += 1
        return new_position

    def __call__(self, E):
        from ..utilities import myassert
        if self.s is None:
            self.s = E.gradient ** 2
        # FIXME come up with somthing how to determine convergence
        convergence = 0
        for i in range(self.steps):
            x = self._step(E.position, E.gradient)
            E = E.resample_at(x)
            myassert(isinstance(E, Energy))
            myassert(x.domain is E.position.domain)
        return E, convergence

    def reset(self):
        self.counter = 1
        self.s = None
