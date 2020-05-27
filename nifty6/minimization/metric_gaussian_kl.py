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


class _KLMetric(EndomorphicOperator):
    def __init__(self, KL):
        self._KL = KL
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = KL.position.domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._KL.apply_metric(x)

    def draw_sample(self, from_inverse=False):
        return self._KL._metric_sample(from_inverse)


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
    nanisinf : bool
        If true, nan energies which can happen due to overflows in the forward
        model are interpreted as inf. Thereby, the code does not crash on
        these occaisions but rather the minimizer is told that the position it
        has tried is not sensible.
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
                 nanisinf=False):
        super(MetricGaussianKL, self).__init__(mean)

        if not isinstance(hamiltonian, StandardHamiltonian):
            raise TypeError
        if hamiltonian.domain is not mean.domain:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        self._constants = tuple(constants)
        self._point_estimates = tuple(point_estimates)
        self._mitigate_nans = nanisinf
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
                    _local_samples.append(met.draw_sample(from_inverse=True))
            _local_samples = tuple(_local_samples)
        else:
            if len(_local_samples) != self._hi-self._lo:
                raise ValueError("# of samples mismatch")
        self._local_samples = _local_samples
        self._lin = Linearization.make_partial_var(mean, self._constants)
        v, g = [], []
        for s in self._locsamp:
            tmp = self._hamiltonian(self._lin+s)
            v.append(tmp.val.val_rw())
            g.append(tmp.gradient)
        self._val = self._sumup(v)[()]/self._n_eff_samples
        if np.isnan(self._val) and self._mitigate_nans:
            self._val = np.inf
        self._grad = self._sumup(g)/self._n_eff_samples
        self._metric = None

    def at(self, position):
        return MetricGaussianKL(
            position, self._hamiltonian, self._n_samples, self._constants,
            self._point_estimates, self._mirror_samples, comm=self._comm,
            _local_samples=self._local_samples, nanisinf=self._mitigate_nans)

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
        lin = self._lin.with_want_metric()
        res = []
        for s in self._locsamp:
            res.append(self._hamiltonian(lin+s).metric(x))
        return self._sumup(res)/self._n_eff_samples

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

    def _sumup(self, obj):
        res = None
        if self._comm is None:
            for o in obj:
                res = o if res is None else res + o
        else:
            ntask = self._comm.Get_size()
            rank = self._comm.Get_rank()
            rank_lo_hi = [_shareRange(self._n_samples, ntask, i) for i in range(ntask)]
            for itask, (l, h) in enumerate(rank_lo_hi):
                for i in range(l, h):
                    iloc = i-self._lo
                    if self._mirror_samples:
                        o = obj[2*iloc] if rank == itask else None
                        o = self._comm.bcast(o, root=itask)
                        res = o if res is None else res + o
                        o = obj[2*iloc+1] if rank == itask else None
                        o = self._comm.bcast(o, root=itask)
                        res = o if res is None else res + o
                    else:
                        o = obj[iloc] if rank == itask else None
                        o = self._comm.bcast(o, root=itask)
                        res = o if res is None else res + o

        return res

    @property
    def _locsamp(self):
        for s in self._local_samples:
            yield s
            if self._mirror_samples:
                yield -s

    def _metric_sample(self, from_inverse=False):
        if from_inverse:
            raise NotImplementedError()
        lin = self._lin.with_want_metric()
        samp = []
        sseq = random.spawn_sseq(self._n_samples)
        for i, v in enumerate(self._local_samples):
            with random.Context(sseq[self._lo+i]):
                samp.append(self._hamiltonian(lin+v).metric.draw_sample(from_inverse=False))
                if self._mirror_samples:
                    samp.append(self._hamiltonian(lin-v).metric.draw_sample(from_inverse=False))
        return self._sumup(samp)/self._n_eff_samples
