from __future__ import absolute_import, division, print_function

from ..compat import *
from .energy import Energy
from ..linearization import Linearization
from .. import utilities
from ..field import Field
from ..multi_field import MultiField

from mpi4py import MPI
import numpy as np
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


class KL_Energy_MPI(Energy):
    def __init__(self,
                 position,
                 h,
                 nsamp,
                 constants=[],
                 constants_samples=None,
                 _samples=None,
                 want_metric=False):
        super(KL_Energy_MPI, self).__init__(position)
        self._h = h
        self._nsamp = nsamp
        self._constants = constants
        if constants_samples is None:
            constants_samples = constants
        self._constants_samples = constants_samples
        self._want_metric = want_metric
        if _samples is None:
            lo, hi = _shareRange(nsamp, ntask, rank)
            met = h(Linearization.make_partial_var(position, constants_samples, True)).metric
            _samples = []
            for i in range(lo, hi):
                np.random.seed(i)
                _samples.append(met.draw_sample(from_inverse=True))
        self._samples = tuple(_samples)
        self._lin = Linearization.make_partial_var(position, constants,
                                                   want_metric)
        if len(self._samples) == 0:  # hack if there are too many MPI tasks
            mymap = map(lambda v: 0*self._h(v), (self._lin,))
        else:
            mymap = map(lambda v: self._h(self._lin + v), self._samples)
        tmp = utilities.my_sum(mymap)*(1./self._nsamp)
        self._val = np_allreduce_sum(tmp.val.local_data)[()]
        self._grad = allreduce_sum_field(tmp.gradient)
        self._metric = tmp.metric

    def at(self, position):
        return KL_Energy_MPI(position, self._h, self._nsamp, self._constants,
                             self._constants_samples, self._samples, self._want_metric)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def apply_metric(self, x):
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


class KL_Energy(Energy):
    def __init__(self, position, h, nsamp, constants=[],
                 constants_samples=None, _samples=None):
        super(KL_Energy, self).__init__(position)
        self._h = h
        self._constants = constants
        if constants_samples is None:
            constants_samples = constants
        self._constants_samples = constants_samples
        if _samples is None:
            met = h(Linearization.make_partial_var(
                position, constants_samples, True)).metric
            _samples = tuple(met.draw_sample(from_inverse=True)
                             for _ in range(nsamp))
        self._samples = _samples

        self._lin = Linearization.make_partial_var(position, constants)
        v, g = None, None
        for s in self._samples:
            tmp = self._h(self._lin+s)
            if v is None:
                v = tmp.val.local_data[()]
                g = tmp.gradient
            else:
                v += tmp.val.local_data[()]
                g = g + tmp.gradient
        self._val = v / len(self._samples)
        self._grad = g * (1./len(self._samples))
        self._metric = None

    def at(self, position):
        return KL_Energy(position, self._h, 0, self._constants,
                         self._constants_samples, self._samples)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def _get_metric(self):
        if self._metric is None:
            lin = self._lin.with_want_metric()
            mymap = map(lambda v: self._h(lin+v).metric, self._samples)
            self._metric = utilities.my_sum(mymap)
            self._metric = self._metric.scale(1./len(self._samples))

    def apply_metric(self, x):
        self._get_metric()
        return self._metric(x)

    @property
    def metric(self):
        self._get_metric()
        return self._metric

    @property
    def samples(self):
        return self._samples
