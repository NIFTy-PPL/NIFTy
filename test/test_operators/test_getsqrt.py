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
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import pytest

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize

dom = ift.RGSpace(5)

@pmp("inp", [(4., True), (-3., False), (2, True), (2.+0j, False), (1, True)])
@pmp("mode", [0, 1, 2])
def test_get_sqrt(inp, mode):
    val, shouldwork = inp
    fld = ift.full(dom, val)
    if mode == 0:
        op = ift.makeOp(fld)
    elif mode == 1:
        op = ift.makeOp(val, ift.makeDomain(dom))
    elif mode == 2:
        fft = ift.FFTOperator(dom)
        if val == 1:
            op = ift.SandwichOperator.make(fft)
        else:
            cheese = ift.makeOp(ift.full(fft.target, val))
            op = ift.SandwichOperator.make(fft, cheese=cheese)
    if not shouldwork:
        with pytest.raises(ValueError):
            op.get_sqrt()
        return
    inp = ift.from_random(op.domain)
    res0 = (op.get_sqrt().adjoint @ op.get_sqrt())(inp)
    res1 = op(inp)
    ift.extra.assert_allclose(res0, res1, rtol=1e-15)
