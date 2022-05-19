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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest
from nifty8 import myassert
from numpy.testing import assert_allclose, assert_raises

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp("scale", [1, 0.5, 0.23, 0.98])
def test_scaled_lh(scale):
    dom = ift.UnstructuredDomain(10)
    d = ift.makeField(dom, (5*ift.from_random(dom)).abs().val.astype(int))
    lh0 = ift.PoissonianEnergy(d) @ ift.Operator.identity_operator(dom).exp()
    lh1 = scale*lh0 + (1-scale)*lh0

    pos = ift.from_random(dom)+1
    pos1 = ift.from_random(dom)

    met0 = lh0.get_metric_at(pos)
    met1 = lh1.get_metric_at(pos)

    ift.extra.assert_allclose(met0(pos1), met1(pos1))
