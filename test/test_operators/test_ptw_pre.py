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
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import pytest

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp('f', [
    'log', 'exp', 'sqrt', 'sin', 'cos', 'tan', 'sinc', 'sinh', 'cosh', 'tanh',
    'absolute', 'reciprocal', 'sigmoid', 'log10', 'log1p', 'expm1', 'softplus',
    ('power', 2.), ('exponentiate', 1.1)
])
def test_ptw_pre(f):
    if not isinstance(f, tuple):
        f = (f,)
    op = ift.FFTOperator(ift.RGSpace(10))
    op0 = op @ ift.ScalingOperator(op.domain, 1.).ptw(*f)
    op1 = op.ptw_pre(*f)
    pos = ift.from_random(op0.domain)
    if f[0] in ['log', 'sqrt', 'log10', 'log1p']:
        pos = pos.exp()
    ift.extra.assert_equal(op0(pos), op1(pos))
