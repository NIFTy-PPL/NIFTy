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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from .. import utilities
from ..linearization import Linearization
from ..operators.energy_operators import StandardHamiltonian
from ..operators.endomorphic_operator import EndomorphicOperator
from .energy import Energy
from mpi4py import MPI
import numpy as np
from ..probing import approximation2endo
from ..sugar import makeOp, full
from ..field import Field
from ..multi_field import MultiField


_comm = MPI.COMM_WORLD
ntask = _comm.Get_size()
rank = _comm.Get_rank()
master = (rank == 0)


def _shareRange(nwork, nshares, myshare):
    nbase = nwork//nshares
    additional = nwork % nshares
    lo = myshare*nbase + min(myshare, additional)
    hi = lo + nbase + int(myshare < additional)
    return lo, hi


def np_allreduce_sum(arr):
    arr = np.array(arr)
    res = np.empty_like(arr)
    _comm.Allreduce(arr, res, MPI.SUM)
    return res


def allreduce_sum_field(fld):
    if isinstance(fld, Field):
        return Field.from_local_data(fld.domain,
                                     np_allreduce_sum(fld.local_data))
    res = tuple(
        Field.from_local_data(f.domain, np_allreduce_sum(f.local_data))
        for f in fld.values())
    return MultiField(fld.domain, res)


class KLMetric(EndomorphicOperator):
    def __init__(self, KL):
        self._KL = KL
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = KL.position.domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._KL.apply_metric(x)

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        return self._KL.metric_sample(from_inverse, dtype)



class MetricGaussianKL_MPI(Energy):
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
    The samples of this class are distributed among MPI tasks.

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
    _samples : None
        Only a parameter for internal uses. Typically not to be set by users.
    seed_offset : int
        A parameter with which one can controll from which seed the samples 
        are drawn. Per default, the seed is different for MPI tasks, but the
        same every time this class is initialized.

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
                 napprox=0, _samples=None, seed_offset=0):
        super(MetricGaussianKL_MPI, self).__init__(mean)

        if not isinstance(hamiltonian, StandardHamiltonian):
            raise TypeError
        if hamiltonian.domain is not mean.domain:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        self._constants = list(constants)
        self._point_estimates = list(point_estimates)
        if not isinstance(mirror_samples, bool):
            raise TypeError

        self._hamiltonian = hamiltonian

        if _samples is None:
            if mirror_samples:
                lo, hi = _shareRange(n_samples*2, ntask, rank)
            else:
                lo, hi = _shareRange(n_samples, ntask, rank)
            met = hamiltonian(Linearization.make_partial_var(
                mean, point_estimates, True)).metric
            if napprox > 1:
                met._approximation = makeOp(approximation2endo(met, napprox))
            _samples = []
            for i in range(lo, hi):
                if mirror_samples:
                    np.random.seed(i//2+seed_offset)
                    if (i % 2) and (i-1 >= lo):
                        _samples.append(-_samples[-1])

                    else:
                        _samples.append(((i % 2)*2-1) *
                                        met.draw_sample(from_inverse=True))
                else:
                    np.random.seed(i)
                    _samples.append(met.draw_sample(from_inverse=True))
            _samples = tuple(_samples)
            if mirror_samples:
                n_samples *= 2
        self._samples = _samples
        self._seed_offset = seed_offset
        self._n_samples = n_samples
        self._lin = Linearization.make_partial_var(mean, constants)
        v, g = None, None
        if len(self._samples) == 0:  # hack if there are too many MPI tasks
            tmp = self._hamiltonian(self._lin)
            v = 0. * tmp.val.local_data
            g = 0. * tmp.gradient
        else:
            for s in self._samples:
                tmp = self._hamiltonian(self._lin+s)
                if v is None:
                    v = tmp.val.local_data.copy()
                    g = tmp.gradient
                else:
                    v += tmp.val.local_data
                    g = g + tmp.gradient
        self._val = np_allreduce_sum(v)[()] / self._n_samples
        self._grad = allreduce_sum_field(g) / self._n_samples
        self._metric = None

    def at(self, position):
        return MetricGaussianKL_MPI(
            position, self._hamiltonian, self._n_samples, self._constants,
            self._point_estimates, _samples=self._samples,
            seed_offset=self._seed_offset)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def _get_metric(self):
        lin = self._lin.with_want_metric()
        if self._metric is None:
            if len(self._samples) == 0:  # hack if there are too many MPI tasks
                self._metric = self._hamiltonian(lin).metric.scale(0.)
            else:
                mymap = map(lambda v: self._hamiltonian(lin+v).metric,
                            self._samples)
                self.unscaled_metric = utilities.my_sum(mymap)
                self._metric = self.unscaled_metric.scale(1./self._n_samples)

    def apply_metric(self, x):
        self._get_metric()
        return allreduce_sum_field(self._metric(x))

    @property
    def metric(self):
        return KLMetric(self)

    @property
    def samples(self):
        res = _comm.allgather(self._samples)
        res = [item for sublist in res for item in sublist]
        return res

    def unscaled_metric_sample(self, from_inverse=False, dtype=np.float64):
        if from_inverse:
            raise NotImplementedError()
        lin = self._lin.with_want_metric()
        samp = full(self._hamiltonian.domain, 0.)
        for v in self._samples:
            samp = samp + self._hamiltonian(lin+v).metric.draw_sample(from_inverse=False, dtype=dtype)
        return allreduce_sum_field(samp)

    def metric_sample(self, from_inverse=False, dtype=np.float64):
        return self.unscaled_metric_sample(from_inverse, dtype)/self._n_samples
