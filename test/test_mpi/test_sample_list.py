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

import nifty8 as ift
import pytest
from mpi4py import MPI

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize
comm = list2fixture([MPI.COMM_WORLD, None])


def _get_sample_list(communicator, cls):
    dom = ift.makeDomain({"a": ift.UnstructuredDomain(2), "b": ift.RGSpace(12)})
    samples = [ift.from_random(dom) for _ in range(3)]
    if cls == "SampleList":
        return ift.SampleList(samples, communicator), samples
    elif cls == "ResidualSampleList":
        mean = ift.from_random(dom)
        neg = 3*[False]
        return ift.ResidualSampleList(mean, samples, neg, communicator), [mean + ss for ss in samples]
    raise NotImplementedError


def _get_ops(sample_list):
    dom = sample_list.domain
    return [None,
            ift.ScalingOperator(dom, 1.),
            ift.ducktape(None, dom, "a") @ ift.ScalingOperator(dom, 1.).exp()]


def test_sample_list(comm):
    sl, samples = _get_sample_list(comm, "SampleList")
    assert comm == sl.comm

    for op in _get_ops(sl):
        sc = ift.StatCalculator()
        if op is None:
            [sc.add(ss) for ss in samples]
        else:
            [sc.add(op(ss)) for ss in samples]
        mean, var = sl.sample_stat(op)
        ift.extra.assert_allclose(mean, sl.average(op))
        ift.extra.assert_allclose(mean, sc.mean)  # FIXME Why does this not fail for comm != None?
        if comm is None:
            ift.extra.assert_allclose(var, sc.var)

        samples = list(samples)
        if op is not None:
            samples = [op(ss) for ss in samples]

        for s0, s1 in zip(samples, sl.iterator(op)):
            ift.extra.assert_equal(s0, s1)

        if comm is None:
            assert len(samples) == sl.n_samples()
        else:
            assert len(samples) <= sl.n_samples()


@pmp("cls", ["ResidualSampleList", "SampleList"])
def test_load_and_save(comm, cls):
    sl, _ = _get_sample_list(comm, cls)

    if comm is None and ift.utilities.get_MPI_params()[1] > 1:
        with pytest.raises(RuntimeError):
            sl.save("sl")
        return

    sl.save("sl")
    sl1 = getattr(ift, cls).load("sl", comm)

    for s0, s1 in zip(sl.local_iterator(), sl1.local_iterator()):
        ift.extra.assert_equal(s0, s1)

    for s0, s1 in zip(sl.local_iterator(), sl1.local_iterator()):
        ift.extra.assert_equal(s0, s1)


@pmp("cls", ["ResidualSampleList", "SampleList"])
@pmp("mean", [False, True])
@pmp("std", [False, True])
@pmp("samples", [False, True])
def test_save_to_hdf5(comm, cls, mean, std, samples):
    pytest.importorskip("h5py")
    sl, _ = _get_sample_list(comm, cls)
    for op in _get_ops(sl):
        if comm is None and ift.utilities.get_MPI_params()[1] > 1:
            with pytest.raises(RuntimeError):
                sl.save_to_hdf5("output.h5", op, mean=mean, std=std, samples=samples)
            continue

        if not mean and not std and not samples:
            with pytest.raises(ValueError):
                sl.save_to_hdf5("output.h5", op, mean=mean, std=std, samples=samples)
            continue

        sl.save_to_hdf5("output.h5", op, mean=mean, std=std, samples=samples, overwrite=True)

        if comm is not None:
            comm.Barrier()
