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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest

from ..common import setup_function, teardown_function


@pytest.mark.parametrize('sp', [
    ift.RGSpace(4),
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.LMSpace(5),
    ift.HPSpace(4),
    ift.GLSpace(4)
])
@pytest.mark.parametrize('seed', [13, 2])
def test_value_inserter(sp, seed):
    with ift.random.Context(seed):
        ind = tuple([int(ift.random.current_rng().integers(0, ss - 1)) for ss in sp.shape])
        op = ift.ValueInserter(sp, ind)
        f = ift.from_random(op.domain, 'normal')
    inp = f.val
    ret = op(f).val
    ift.myassert(ret[ind] == inp)
    ift.myassert(np.sum(ret) == inp)
