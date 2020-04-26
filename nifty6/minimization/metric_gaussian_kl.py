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

from .. import random, utilities
from ..field import Field
from ..linearization import Linearization
from ..multi_field import MultiField
from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.energy_operators import StandardHamiltonian
from ..probing import approximation2endo
from ..sugar import full, makeOp
from .energy import Energy


def _shareRange(nwork, nshares, myshare):
    nbase = nwork//nshares
    additional = nwork % nshares
    lo = myshare*nbase + min(myshare, additional)
    hi = lo + nbase + int(myshare < additional)
    return lo, hi


def _np_allreduce_sum(comm, arr):
    if comm is None:
        return arr
    from mpi4py import MPI
    arr = np.array(arr)
    res = np.empty_like(arr)
    comm.Allreduce(arr, res, MPI.SUM)
    return res


def _allreduce_sum_field(comm, fld):
    if comm is None:
        return fld
    if isinstance(fld, Field):
        return Field(fld.domain, _np_allreduce_sum(fld.val))
    res = tuple(
        Field(f.domain, _np_allreduce_sum(comm, f.val))
        for f in fld.values())
    return MultiField(fld.domain, res)


class _KLMetric(EndomorphicOperator):
    def __init__(self, KL):
        self._KL = KL
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = KL.position.domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._KL.apply_metric(x)

    def draw_sample(self, dtype, from_inverse=False):
        return self._KL._metric_sample(dtype, from_inverse)


class MetricGaussianKL(Energy):
    """Provides the sampled Kullback-Leibler divergence between a distribution
    and a Metric Gaussian.

    A Metric Gaussian is used to approximate another probability distribution.
    It is a Gaussian distribution that uses the Fisher information metric of
    the other distribution at the location of its mean to approximate the
    variance. In order to infer the mean, a stochastic estimate of the
    Kullback-Leibler divergence is minimized. This estimate is obtained by
    sampling the Metric Gaussian at the current mean. During minimization
    these samples are kept constant; only the mean is updated. Due to the
    typically nonlinear structure of the true distribution these samples have
    to be updated eventually by intantiating `MetricGaussianKL` again. For the
    true probability distribution the standard parametrization is assumed.
    The samples of this class can be distributed among MPI tasks.

    Parameters
    ----------
    mean : Field
        Mean of the Gaussian probability distribution.
    hamiltonian : StandardHamiltonian
        Hamiltonian of the approximated probability distribution.
    n_samples : integer
        Number of samples used to stochastically estimate the KL.
    constants : list
        List of parameter keys that are kept constant during optimization.
        Default is no constants.
    point_estimates : list
        List of parameter keys for which no samples are drawn, but that are
        (possibly) optimized for, corresponding to point estimates of these.
        Default is to draw samples for the complete domain.
    mirror_samples : boolean
        Whether the negative of the drawn samples are also used,
        as they are equally legitimate samples. If true, the number of used
        samples doubles. Mirroring samples stabilizes the KL estimate as
        extreme sample variation is counterbalanced. Default is False.
    napprox : int
        Number of samples for computing preconditioner for sampling. No
        preconditioning is done by default.
    comm : MPI communicator or None
        If not None, samples will be distributed as evenly as possible
        across this communicator. If `mirror_samples` is set, then a sample and
        its mirror image will always reside on the same task.
    lh_sampling_dtype : type
        Determines which dtype in data space shall be used for drawing samples
        from the metric. If the inference is based on complex data,
        lh_sampling_dtype shall be set to complex accordingly. The reason for
        the presence of this parameter is that metric of the likelihood energy
        is just an `Operator` which does not know anything about the dtype of
        the fields on which it acts. Default is float64.
    _local_samples : None
        Only a parameter for internal uses. Typically not to be set by users.

    Note
    ----
    The two lists `constants` and `point_estimates` are independent from each
    other. It is possible to sample along domains which are kept constant
    during minimization and vice versa.

    See also
    --------
    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """

    def __init__(self, mean, hamiltonian, n_samples, constants=[],
                 point_estimates=[], mirror_samples=False,
                 napprox=0, comm=None, _local_samples=None,
                 lh_sampling_dtype=np.float64):
        super(MetricGaussianKL, self).__init__(mean)

        if not isinstance(hamiltonian, StandardHamiltonian):
            raise TypeError
        if hamiltonian.domain is not mean.domain:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        self._constants = tuple(constants)
        self._point_estimates = tuple(point_estimates)
        if not isinstance(mirror_samples, bool):
            raise TypeError

        self._hamiltonian = hamiltonian

        self._n_samples = int(n_samples)
        if comm is not None:
            self._comm = comm
            ntask = self._comm.Get_size()
            rank = self._comm.Get_rank()
            self._lo, self._hi = _shareRange(self._n_samples, ntask, rank)
        else:
            self._comm = None
            self._lo, self._hi = 0, self._n_samples

        self._mirror_samples = bool(mirror_samples)
        self._n_eff_samples = self._n_samples
        if self._mirror_samples:
            self._n_eff_samples *= 2

        if _local_samples is None:
            met = hamiltonian(Linearization.make_partial_var(
                mean, self._point_estimates, True)).metric
            if napprox >= 1:
                met._approximation = makeOp(approximation2endo(met, napprox))
            _local_samples = []
            sseq = random.spawn_sseq(self._n_samples)
            for i in range(self._lo, self._hi):
                with random.Context(sseq[i]):
                    _local_samples.append(met.draw_sample(
                        dtype=lh_sampling_dtype, from_inverse=True))
            _local_samples = tuple(_local_samples)
        else:
            if len(_local_samples) != self._hi-self._lo:
                raise ValueError("# of samples mismatch")
        self._local_samples = _local_samples
        self._lin = Linearization.make_partial_var(mean, self._constants)
        v, g = None, None
        if len(self._local_samples) == 0:  # hack if there are too many MPI tasks
            tmp = self._hamiltonian(self._lin)
            v = 0. * tmp.val.val
            g = 0. * tmp.gradient
        else:
            for s in self._local_samples:
                tmp = self._hamiltonian(self._lin+s)
                if self._mirror_samples:
                    tmp = tmp + self._hamiltonian(self._lin-s)
                if v is None:
                    v = tmp.val.val_rw()
                    g = tmp.gradient
                else:
                    v += tmp.val.val
                    g = g + tmp.gradient
        self._val = _np_allreduce_sum(self._comm, v)[()] / self._n_eff_samples
        self._grad = _allreduce_sum_field(self._comm, g) / self._n_eff_samples
        self._metric = None
        self._sampdt = lh_sampling_dtype

    def at(self, position):
        return MetricGaussianKL(
            position, self._hamiltonian, self._n_samples, self._constants,
            self._point_estimates, self._mirror_samples, comm=self._comm,
            _local_samples=self._local_samples, lh_sampling_dtype=self._sampdt)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def _get_metric(self):
        lin = self._lin.with_want_metric()
        if self._metric is None:
            if len(self._local_samples) == 0:  # hack if there are too many MPI tasks
                self._metric = self._hamiltonian(lin).metric.scale(0.)
            else:
                mymap = map(lambda v: self._hamiltonian(lin+v).metric,
                            self._local_samples)
                unscaled_metric = utilities.my_sum(mymap)
                if self._mirror_samples:
                    mymap = map(lambda v: self._hamiltonian(lin-v).metric,
                            self._local_samples)
                    unscaled_metric = unscaled_metric + utilities.my_sum(mymap)
                self._metric = unscaled_metric.scale(1./self._n_eff_samples)

    def apply_metric(self, x):
        self._get_metric()
        return _allreduce_sum_field(self._comm, self._metric(x))

    @property
    def metric(self):
        return _KLMetric(self)

    @property
    def samples(self):
        if self._comm is None:
            for s in self._local_samples:
                yield s
                if self._mirror_samples:
                    yield -s
        else:
            ntask = self._comm.Get_size()
            rank = self._comm.Get_rank()
            rank_lo_hi = [_shareRange(self._n_samples, ntask, i) for i in range(ntask)]
            for itask, (l, h) in enumerate(rank_lo_hi):
                for i in range(l, h):
                    data = self._local_samples[i-self._lo] if rank == itask else None
                    s = self._comm.bcast(data, root=itask)
                    yield s
                    if self._mirror_samples:
                        yield -s

    def _metric_sample(self, dtype, from_inverse=False):
        if from_inverse:
            raise NotImplementedError()
        lin = self._lin.with_want_metric()
        samp = full(self._hamiltonian.domain, 0.)
        sseq = random.spawn_sseq(self._n_samples)
        for i, v in enumerate(self._local_samples):
            with random.Context(sseq[self._lo+i]):
                samp = samp + self._hamiltonian(lin+v).metric.draw_sample(dtype=dtype, from_inverse=False)
                if self._mirror_samples:
                    samp = samp + self._hamiltonian(lin-v).metric.draw_sample(dtype=dtype, from_inverse=False)
        return _allreduce_sum_field(self._comm, samp)/self._n_eff_samples
