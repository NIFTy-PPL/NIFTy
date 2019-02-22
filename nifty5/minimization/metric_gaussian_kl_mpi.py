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
from .energy import Energy
from mpi4py import MPI
import numpy as np
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


class MetricGaussianKL_MPI(Energy):
    def __init__(self, mean, hamiltonian, n_samples, constants=[],
                 point_estimates=[], mirror_samples=False,
                 _samples=None):
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
            lo, hi = _shareRange(n_samples, ntask, rank)
            met = hamiltonian(Linearization.make_partial_var(
                mean, point_estimates, True)).metric
            _samples = []
            for i in range(lo, hi):
                np.random.seed(i)
                _samples.append(met.draw_sample(from_inverse=True))
            if mirror_samples:
                _samples += [-s for s in _samples]
                n_samples *= 2
            _samples = tuple(_samples)
        self._samples = _samples
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
            self._point_estimates, _samples=self._samples)

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
                self._metric = utilities.my_sum(mymap)
                self._metric = self._metric.scale(1./self._n_samples)

    def apply_metric(self, x):
        self._get_metric()
        return allreduce_sum_field(self._metric(x))

    @property
    def metric(self):
        if ntask > 1:
            raise ValueError("not supported when MPI is active")
        return self._metric

    @property
    def samples(self):
        res = _comm.allgather(self._samples)
        res = [item for sublist in res for item in sublist]
        return res
