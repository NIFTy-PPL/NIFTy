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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from glob import glob
from os import remove

import pytest


def list2fixture(lst):
    @pytest.fixture(params=lst)
    def myfixture(request):
        return request.param

    return myfixture


def setup_function():
    import nifty8 as ift
    ift.random.push_sseq_from_seed(42)
    comm, _, _, master = ift.utilities.get_MPI_params()
    if master:
        for ff in glob("*.pickle") + glob("*.png") + glob("*.h5"):
            remove(ff)
    if comm is not None:
        comm.Barrier()


def teardown_function():
    import nifty8 as ift
    ift.random.pop_sseq()
