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

from .. import random
from ..domain_tuple import DomainTuple
from ..linearization import Linearization
from ..minimization.energy import Energy
from ..multi_domain import MultiDomain
from ..sugar import from_random
from ..utilities import (allreduce_sum, get_MPI_params_from_comm, myassert,
                         shareRange)


class EnergyAdapter(Energy):
    """Helper class which provides the traditional Nifty Energy interface to
    Nifty operators with a scalar target domain.

    Parameters
    -----------
    position : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField`
        The position where the minimization process is started.
    op : EnergyOperator
        The expression computing the energy from the input data.
    constants : list of strings
        The component names of the operator's input domain which are assumed
        to be constant during the minimization process.
        If the operator's input domain is not a MultiField, this must be empty.
        Default: [].
    want_metric : bool
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
        if len(constants) > 0:
            cstpos = position.extract_by_keys(constants)
            _, op = op.simplify_for_constant_input(cstpos)
            varkeys = set(op.domain.keys()) - set(constants)
            position = position.extract_by_keys(varkeys)
        super(EnergyAdapter, self).__init__(position)
        self._op = op
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


class StochasticEnergyAdapter(Energy):
    """Provide the energy interface for an energy operator where parts of the
    input are averaged instead of optimized.

    Specifically, a set of standard normal distributed samples are drawn for
    the input corresponding to `keys` and each sample is inserted partially
    into `op`. The resulting operators are then averaged.  The subdomain that
    is not sampled is left a stochastic average of an energy with the remaining
    subdomain being the DOFs that are considered to be optimization parameters.

    Notes
    -----
    `StochasticEnergyAdapter` should never be created using the constructor,
    but rather via the factory function :attr:`make`.
    """
    def __init__(self, position, op, keys, local_ops, n_samples, comm, nanisinf,
                 noise, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(StochasticEnergyAdapter, self).__init__(position)
        for lop in local_ops:
            myassert(position.domain == lop.domain)
        self._comm = comm
        self._local_ops = local_ops
        self._n_samples = n_samples
        self._nanisinf = nanisinf
        lin = Linearization.make_var(position)
        v, g = [], []
        for lop in self._local_ops:
            tmp = lop(lin)
            v.append(tmp.val.val)
            g.append(tmp.gradient)
        self._val = allreduce_sum(v, self._comm)[()]/self._n_samples
        if np.isnan(self._val) and self._nanisinf:
            self._val = np.inf
        self._grad = allreduce_sum(g, self._comm)/self._n_samples
        self._noise = noise

        self._op = op
        self._keys = keys

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def at(self, position):
        return StochasticEnergyAdapter(position, self._op, self._keys,
                    self._local_ops, self._n_samples, self._comm, self._nanisinf,
                    self._noise, _callingfrommake=True)

    def apply_metric(self, x):
        lin = Linearization.make_var(self.position, want_metric=True)
        res = []
        for op in self._local_ops:
            res.append(op(lin).metric(x))
        return allreduce_sum(res, self._comm)/self._n_samples

    @property
    def metric(self):
        from .kl_energies import _SelfAdjointOperatorWrapper
        return _SelfAdjointOperatorWrapper(self.position.domain,
                                           self.apply_metric)

    def resample_at(self, position):
        return StochasticEnergyAdapter.make(position, self._op, self._keys,
                                            self._n_samples, self._comm)

    @staticmethod
    def make(position, op, sampling_keys, n_samples, mirror_samples,
             comm=None, nanisinf=False):
        """Factory function for StochasticEnergyAdapter.

        Parameters
        ----------
        position : :class:`nifty8.multi_field.MultiField`
            Values of the optimization parameters
        op : Operator
            The objective function of the optimization problem. Must have a
            scalar target. The domain must be a `MultiDomain` with its keys
            being the union of `sampling_keys` and `position.domain.keys()`.
        sampling_keys : iterable of String
            The keys of the subdomain over which the stochastic average of `op`
            should be performed.
        n_samples : int
            Number of samples used for the stochastic estimate.
        mirror_samples : boolean
            Whether the negative of the drawn samples are also used, as they are
            equally legitimate samples. If true, the number of used samples
            doubles.
        comm : MPI communicator or None
            If not None, samples will be distributed as evenly as possible
            across this communicator. If `mirror_samples` is set, then a sample
            and its mirror image will always reside on the same task.
        nanisinf : bool
            If true, nan energies, which can occur due to overflows in the
            forward model, are interpreted as inf which can be interpreted by
            optimizers.
        """
        myassert(op.target == DomainTuple.scalar_domain())
        samdom = {}
        if not isinstance(n_samples, int):
            raise TypeError
        for k in sampling_keys:
            if (k in position.domain.keys()) or (k not in op.domain.keys()):
                raise ValueError
            samdom[k] = op.domain[k]
        samdom = MultiDomain.make(samdom)
        noise = []
        sseq = random.spawn_sseq(n_samples)

        ntask, rank, _ = get_MPI_params_from_comm(comm)
        for i in range(*shareRange(n_samples, ntask, rank)):
            with random.Context(sseq[i]):
                rnd = from_random(samdom)
                noise.append(rnd)
                if mirror_samples:
                    noise.append(-rnd)
        local_ops = []
        for nn in noise:
            _, tmp = op.simplify_for_constant_input(nn)
            myassert(tmp.domain == position.domain)
            local_ops.append(tmp)
        n_samples = 2*n_samples if mirror_samples else n_samples
        return StochasticEnergyAdapter(position, op, sampling_keys, local_ops,
                              n_samples, comm, nanisinf, noise, _callingfrommake=True)

    def samples(self):
        return self._noise
