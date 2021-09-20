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
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras, Philipp Frank

import pickle

from .. import utilities

from ..multi_domain import MultiDomain


class SampleList:
    def __init__(self, comm, domain):
        from ..sugar import makeDomain
        self._comm = comm
        self._domain = makeDomain(domain)
        utilities.check_MPI_equality(self._domain, comm)

    @staticmethod
    def all_indices(n_samples, comm):
        ntask, rank, _ = utilities.get_MPI_params_from_comm(comm)
        return [range(*utilities.shareRange(n_samples, ntask, i)) for i in range(ntask)]

    @property
    def comm(self):
        return self._comm

    @property
    def domain(self):
        return self._domain

    def global_sample_iterator(self, op=None):
        op = _none_to_id(op)
        if self.comm is not None:
            for itask in range(self.comm.get_Size()):
                for i in range(_bcast(len(self), self._comm, itask)):
                    ss = _bcast(self[i], self._comm, itask)
                    yield op(ss)
        return (op(ss) for ss in self)

    def global_average(self, op=None):
        """if op returns tuple, then individual averages are computed and returned individually."""
        op = _none_to_id(op)
        res = [op(ss) for ss in self]
        n = self.global_n_samples()
        if not isinstance(res[0], tuple):
            return utilities.allreduce_sum(res, self.comm) / n
        res = [[elem[ii] for elem in res] for ii in range(len(res[0]))]
        return tuple(utilities.allreduce_sum(rr, self.comm)/n for rr in res)

    def global_n_samples(self):
        return utilities.allreduce_sum([len(self)], self.comm)

    def __len__(self):
        """Local length"""
        raise NotImplementedError

    def global_sample_stat(self, op):
        from ..probing import StatCalculator
        sc = StatCalculator()
        for ss in self.global_sample_iterator(op):
            sc.add(ss)
        return sc.mean, sc.var

    def global_mean(self, op=None):
        return self.global_sample_stat(op)[0]

    def global_sd(self, op=None):
        return self.global_sample_stat(op)[1].sqrt()

    def save(self, file_name_base):
        if self._comm is not None:
            raise NotImplementedError
        with open(file_name_base + ".pickle", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name_base, comm=None):
        if comm is not None:
            raise NotImplementedError
        with open(file_name_base + ".pickle", "rb") as f:
            obj = pickle.load(f)
        utilities.myassert(isinstance(obj, SampleList))
        return obj


class ResidualSampleList(SampleList):
    def __init__(self, mean, residuals, neg, comm):
        """
        Entries in dict of residual can be missing -> no residual is added
        """
        super(ResidualSampleList, self).__init__(comm, mean.domain)
        self._m = mean
        self._r = tuple(residuals)
        self._n = tuple(neg)

        assert len(self._r) == len(self._n)
        r_dom = self._r[0].domain
        assert isinstance(r_dom, MultiDomain)
        assert all(rr.domain is r_dom for rr in self._r)
        assert all(k in self._m.domain.keys() for k in r_dom.keys())
        n_keys = self._n[0].keys()
        assert all(elem.keys() == n_keys for elem in neg)
        assert n_keys == r_dom.keys()
        assert all(isinstance(xx, bool) for elem in neg for xx in elem)  # FIXME is this the correct order?

    @property
    def mean(self):
        return self._m

    def at(self, mean):
        return ResidualSampleList(mean, self._r, self._n, self.comm)

    def __getitem__(self, i):
        return self._m.flexible_addsub(self._r[i], self._n[i])

    def __len__(self):
        return len(self._r)


def _none_to_id(obj):
    if obj is None:
        return lambda x: x
    return obj


def _bcast(obj, comm, root):
    data = obj if comm.Get_rank() == root else None
    return comm.bcast(data, root=root)
