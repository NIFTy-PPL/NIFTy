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
# Author: Philipp Arras

import pickle


class SampleList:
    def __init__(self, comm, domain):
        from ..sugar import makeDomain
        from ..utilities import check_MPI_equality
        self._comm = comm
        self._domain = makeDomain(domain)
        check_MPI_equality(self._domain, comm)

    # def __iter__(self):
    #     raise NotImplementedError

    @property
    def comm(self):
        return self._comm

    @property
    def domain(self):
        return self._domain

    def global_sample_iterator(self, op=None):
        op = _none_to_id(op)
        if self.comm is not None:
            raise NotImplementedError
        return (op(ss) for ss in self)

    def global_n_samples(self):
        from ..utilities import allreduce_sum
        return allreduce_sum([len(self)], self.comm)

    def global_sample_stat(self, op):
        from ..probing import StatCalculator
        from ..operators.scaling_operator import ScalingOperator

        op = _none_to_id(op)

        if self.comm is not None:
            raise NotImplementedError

        sc = StatCalculator()
        for ss in self:
            sc.add(op(ss))
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
        assert isinstance(obj, SampleList)
        return obj


def _none_to_id(obj):
    if obj is None:
        return lambda x: x
    return obj
