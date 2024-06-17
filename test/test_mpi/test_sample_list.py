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
comm = [MPI.COMM_WORLD]
if MPI.COMM_WORLD.Get_size() == 1:
    comm += [None]
comm = list2fixture(comm)


def _get_sample_list(communicator, cls):
    dom = ift.makeDomain({"a": ift.UnstructuredDomain(2), "b": ift.RGSpace(12)})
    samples = [ift.from_random(dom) for _ in range(3)]
    mean = ift.from_random(dom)
    if cls == "SampleList":
        return ift.SampleList(samples, communicator), samples
    elif cls == "ResidualSampleList":
        neg = 3*[False]
        return ift.ResidualSampleList(mean, samples, neg, communicator), [mean + ss for ss in samples]
    elif cls == "SymmetricalSampleList":
        neg = len(samples)*[False] + len(samples)*[True]
        reference = [mean + ss for ss in samples] + [mean - ss for ss in samples]
        samples = samples + samples
        return ift.ResidualSampleList(mean, samples, neg, communicator), reference
    elif cls == "PartiallyEmptyResidualSampleList":
        if communicator is None or communicator.Get_rank() == 0:
            neg = [False]
            resi = samples[0:1]
        else:
            neg = resi = []
        return ift.ResidualSampleList(mean, resi, neg, communicator), [mean + samples[0]]
    elif cls == "PartiallyEmptySampleList":
        if communicator is None or communicator.Get_rank() == 0:
            local_samples = samples[0:1]
        else:
            local_samples = []
        return ift.SampleList(local_samples, comm=communicator, domain=samples[0].domain), samples[0:1]
    raise NotImplementedError


all_cls = ["ResidualSampleList", "SampleList", "PartiallyEmptyResidualSampleList",
           "PartiallyEmptySampleList"]


def _get_ops(sample_list):
    dom = sample_list.domain
    return [None,
            ift.ScalingOperator(dom, 1.),
            ift.ducktape(None, dom, "a") @ ift.ScalingOperator(dom, 1.).exp()]


@pmp("cls", ["SampleList", "PartiallyEmptySampleList"])
def test_sample_list(comm, cls):
    sl, samples = _get_sample_list(comm, cls)
    assert comm == sl.comm

    for op in _get_ops(sl):
        sc = ift.StatCalculator()
        if op is None:
            [sc.add(ss) for ss in samples]
        else:
            [sc.add(op(ss)) for ss in samples]
        if sl.n_samples > 1:
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
            assert len(samples) == sl.n_samples
        else:
            assert len(samples) <= sl.n_samples


@pmp("cls", all_cls)
def test_load_and_save(comm, cls):
    if comm is None and ift.utilities.get_MPI_params()[1] > 1:
        pytest.skip()

    sl, _ = _get_sample_list(comm, cls)
    sl.save("sl")
    lcls = cls[14:] if cls[:14] == "PartiallyEmpty" else cls
    sl1 = getattr(ift, lcls).load("sl", comm)

    for s0, s1 in zip(sl.local_iterator(), sl1.local_iterator()):
        ift.extra.assert_equal(s0, s1)

    for s0, s1 in zip(sl.local_iterator(), sl1.local_iterator()):
        ift.extra.assert_equal(s0, s1)


def test_load_mean(comm):
    sl, _ = _get_sample_list(comm, "SymmetricalSampleList")
    m0 = sl._m
    sl.save("sl")
    sl1 = ift.ResidualSampleList.load("sl", comm=comm)
    m1, _ = sl1.sample_stat(None)
    m2 = ift.ResidualSampleList.load_mean("sl")
    ift.extra.assert_equal(m0, m2)
    ift.extra.assert_allclose(m1, m2)


@pmp("cls", all_cls)
@pmp("mean", [False, True])
@pmp("std", [False, True])
@pmp("samples", [False, True])
def test_save_to_hdf5(comm, cls, mean, std, samples):
    pytest.importorskip("h5py")
    import h5py
    import numpy as np
    from nifty8 import DomainTuple, MultiDomain, RGSpace, UnstructuredDomain

    if comm is None and ift.utilities.get_MPI_params()[1] > 1:
        pytest.skip()
    sl, _ = _get_sample_list(comm, cls)
    if sl.n_samples < 2 and std:
        pytest.skip()
    for op in _get_ops(sl):
        if not mean and not std and not samples:
            with pytest.raises(ValueError):
                sl.save_to_hdf5("output.h5", op, mean=mean, std=std, samples=samples)
            continue
        sl.save_to_hdf5("output.h5", op, mean=mean, std=std, samples=samples, overwrite=True)
        if comm is not None:
            comm.Barrier()

        flddom = sl.domain if op is None else op.target
        mdom = isinstance(flddom, ift.MultiDomain)

        shps = []
        f = h5py.File("output.h5", "r")
        domain_repr = f.attrs["nifty domain"]
        if mean:
            shps.append(_get_shape(f["stats"]["mean"], mdom))
        if std:
            shps.append(_get_shape(f["stats"]["standard deviation"], mdom))
        if samples:
            for ii in range(sl.n_samples):
                shps.append(_get_shape(f["samples"][str(ii)], mdom))
        assert all(elem == shps[0] for elem in shps)

        dom1 = eval(domain_repr)
        assert dom1 is flddom

        f.close()


def _get_shape(inp, mdom):
    if not mdom:
        return inp.shape
    return {kk: _get_shape(vv, False) for kk, vv in inp.items()}
