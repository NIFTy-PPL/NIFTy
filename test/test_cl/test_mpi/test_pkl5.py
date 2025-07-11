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
# Copyright(C) 2025 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty.cl as ift
import numpy as np
import pytest
from mpi4py import MPI
from mpi4py.util import pkl5
from numpy.testing import assert_allclose

from ..common import setup_function, teardown_function

comm = pkl5.Intracomm(MPI.COMM_WORLD)
rank = comm.Get_rank()
ntasks = comm.Get_size()
mpi = ntasks > 1

size_in_gb = 2.2
npix = int(size_in_gb/8 * 1e9)

pms = pytest.mark.skipif


@pms(not mpi, reason="requires at least two mpi tasks")
def test_MPI_large_object_transfer():
    v = np.full(npix, fill_value=ntasks) if rank == 0 else None
    v = comm.bcast(v, root=0)
    assert_allclose(v, ntasks)


@pms(not mpi, reason="requires at least two mpi tasks")
def test_MPI_large_object_SampleList():
    dom = ift.UnstructuredDomain([npix])
    f = ift.full(dom, rank)
    av = ift.SampleList([f], comm=comm).average()
    assert av.size == npix
    assert np.unique(av.val).size == 1
    assert av.val[0] == np.mean(np.arange(ntasks))
