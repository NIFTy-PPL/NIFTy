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


import numpy as np
import pytest

import nifty8 as ift

from .common import list2fixture, setup_function, teardown_function



class MinimalSampleList(ift.SampleList):
    def __init__(self, samples, comm):
        super(MinimalSampleList, self).__init__(comm, samples[0].domain)
        self._s = samples

    def __getitem__(self, x):
        return self._s[x]

    def __len__(self):
        return len(self._s)


def _get_sample_list(comm):
    dom = ift.makeDomain({"a": ift.UnstructuredDomain(2), "b": ift.RGSpace(12)})
    samples = [ift.from_random(dom) for _ in range(3)]
    return MinimalSampleList(samples, comm), samples


def test_sample_list():
    comm = None
    sl, samples = _get_sample_list(comm)
    dom = sl.domain

    assert comm == sl.comm

    ops = [None, ift.ScalingOperator(dom, 1.),
           ift.ducktape(None, dom, "a") @ ift.ScalingOperator(dom, 1.).exp()]

    for op in ops:
        sc = ift.StatCalculator()
        if op is None:
            [sc.add(ss) for ss in samples]
        else:
            [sc.add(op(ss)) for ss in samples]
        mean, var = sl.global_sample_stat(op)
        print(mean.domain)
        print(sc.mean.domain)
        print(sl.global_mean().domain)
        print()
        ift.extra.assert_allclose(mean, sc.mean)
        ift.extra.assert_allclose(mean, sl.global_mean(op))
        ift.extra.assert_allclose(var, sc.var)
        ift.extra.assert_allclose(var, sl.global_sd(op)**2)

        samples = list(samples)
        if op is not None:
            samples = [op(ss) for ss in samples]

        for s0, s1 in zip(samples, sl.global_iterator(op)):
            ift.extra.assert_equal(s0, s1)
        assert len(samples) == sl.global_n_samples()


def test_load_and_save():
    comm = None

    sl, _ = _get_sample_list(comm)
    sl.save("sample_list")
    sl1 = ift.SampleList.load("sample_list", comm)

    for s0, s1 in zip(sl, sl1):
        ift.extra.assert_equal(s0, s1)

    for s0, s1 in zip(sl, sl1):
        ift.extra.assert_equal(s0, s1)
