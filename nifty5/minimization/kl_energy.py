from __future__ import absolute_import, division, print_function

from ..compat import *
from .energy import Energy
from ..linearization import Linearization
from ..operators.scaling_operator import ScalingOperator
from ..operators.block_diagonal_operator import BlockDiagonalOperator
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


class KL_Energy(Energy):
    def __init__(self,
                 position,
                 h,
                 nsamp,
                 constants=[],
                 _samples=None,
                 want_metric=False):
        super(KL_Energy, self).__init__(position)
        self._h = h
        self._nsamp = nsamp
        self._constants = constants
        self._want_metric = want_metric
        if _samples is None:
            lo, hi = _shareRange(nsamp, ntask, rank)
            met = h(Linearization.make_var(position, True)).metric
            _samples = []
            for i in range(lo, hi):
                np.random.seed(i)
                _samples.append(met.draw_sample(from_inverse=True))
        self._samples = tuple(_samples)
        if len(constants) == 0:
            tmp = Linearization.make_var(position, want_metric)
        else:
            ops = [
                ScalingOperator(0. if key in constants else 1., dom)
                for key, dom in position.domain.items()
            ]
            bdop = BlockDiagonalOperator(position.domain, tuple(ops))
            tmp = Linearization(position, bdop, want_metric=want_metric)
        mymap = map(lambda v: self._h(tmp + v), self._samples)
        tmp = utilities.my_sum(mymap)*(1./self._nsamp)
        self._val = np_allreduce_sum(tmp.val.local_data)[()]
        self._grad = allreduce_sum_field(tmp.gradient)
        self._metric = tmp.metric

    def at(self, position):
        return KL_Energy(position, self._h, self._nsamp, self._constants,
                         self._samples, self._want_metric)

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
