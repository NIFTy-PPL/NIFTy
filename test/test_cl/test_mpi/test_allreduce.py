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
# Copyright(C) 2025 LambdaFields GmbH
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.


import nifty.cl as ift
import numpy as np
import pytest
from mpi4py import MPI
from mpi4py.util import pkl5
from nifty.cl.utilities import allreduce_sum

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize
pms = pytest.mark.skipif


@pmp("input_data", [
    [1, 2, 3, 4],
    [[1, 1], [2, 2], [3, 3], [4, 4]],
])
def test_allreduce_sum_simple_types(input_data):
    result = allreduce_sum(input_data, comm=MPI.COMM_WORLD)

    input_data = MPI.COMM_WORLD.Get_size() * input_data  # All tasks have the same input data
    expected = input_data[0]
    for val in input_data[1:]:
        expected = expected + val

    assert result == expected


dom = ift.RGSpace(2, 0.11)
mdom = ift.makeDomain({"a": ift.RGSpace(2, 0.11), "b": ift.UnstructuredDomain(1)})
@pmp("input_data", [
    [np.array([99.2, 1]), np.array([2, 3])],
    [np.broadcast_to(np.array([[2, 4]]), (2, 2)), np.broadcast_to(np.array([[2, 3]]), (2, 2))],
    [ift.makeField(dom, np.array([99.2, 1])), ift.makeField(dom, np.array([2, 3]))],
    [ift.MultiField.from_raw(mdom, {"a": np.array([99.2, 1]), "b": np.array([12.01])}),
     ift.MultiField.from_raw(mdom, {"a": np.array([81.21, 1.22]), "b": np.array([12.02])})],
])
def test_allreduce_sum_array_types(input_data):
    result = allreduce_sum(input_data, comm=MPI.COMM_WORLD)

    input_data = MPI.COMM_WORLD.Get_size() * input_data  # All tasks have the same input data
    expected = input_data[0]
    for val in input_data[1:]:
        expected = expected + val

    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    elif isinstance(expected, (ift.Field, ift.MultiField)):
        ift.extra.assert_equal(result, expected)
    else:
        assert result == expected


def test_allreduce_sum_parallel():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip(reason="MPI not available")

    # Each process has a single integer, equal to its rank
    local_value = [rank]
    result = allreduce_sum(local_value, comm=comm)
    expected = sum(range(size))

    # All ranks should receive the same final result
    assert result == expected
