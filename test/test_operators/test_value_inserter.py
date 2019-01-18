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

import numpy as np
import pytest
from numpy.testing import assert_allclose

import nifty5 as ift


@pytest.mark.parametrize('sp', [
    ift.RGSpace(4),
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.LMSpace(5),
    ift.HPSpace(4),
    ift.GLSpace(4)
])
@pytest.mark.parametrize('seed', [13, 2])
def test_value_inserter(sp, seed):
    np.random.seed(seed)
    ind = tuple([np.random.randint(0, ss - 1) for ss in sp.shape])
    op = ift.ValueInserter(sp, ind)
    f = ift.from_random('normal', ift.UnstructuredDomain((1,)))
    inp = f.to_global_data()[0]
    ret = op(f).to_global_data()
    assert_allclose(ret[ind], inp)
    assert_allclose(np.sum(ret), inp)


def test_value_inserter_nonzero():
    sp = ift.RGSpace(4)
    ind = (1,)
    default = 1.24
    op = ift.ValueInserter(sp, ind, default)
    f = ift.from_random('normal', ift.UnstructuredDomain((1,)))
    inp = f.to_global_data()[0]
    ret = op(f).to_global_data()
    assert_allclose(ret[ind], inp)
    assert_allclose(np.sum(ret), inp + 3*default)
