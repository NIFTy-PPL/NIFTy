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

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..common import list2fixture, setup_function, teardown_function
from .test_adjoint import _h_spaces, _p_spaces, _pow_spaces

pmp = pytest.mark.parametrize


@pmp("dom0", _h_spaces + _p_spaces + _pow_spaces)
@pmp("dom1", _h_spaces + _p_spaces + _pow_spaces)
def test_multifield2vector(dom0, dom1):
    fld = ift.MultiField.from_dict({'a': ift.from_random(dom0),
                                    'b': ift.from_random(dom1)})
    op = ift.Multifield2Vector(fld.domain)
    ift.extra.assert_allclose(fld, op.adjoint(op(fld)))
    assert op.target.size == fld.size
    
    arr = np.empty(fld.size)
    arr[:dom0.size] = fld['a'].val.flatten()
    arr[dom0.size:] = fld['b'].val.flatten()
    assert_allclose(arr, op(fld).val)
