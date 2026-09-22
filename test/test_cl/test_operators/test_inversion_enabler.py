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
# Copyright(C) 2026 Philipp Arras

import nifty.cl as ift
import numpy as np

from ..common import list2fixture, setup_function, teardown_function

device_id = list2fixture([-1, 0] if ift.device_available() else [-1])

_ic = ift.GradientNormController(iteration_limit=100, tol_abs_gradnorm=1e-10)


def _sum_operator(device_id):
    """Operator without inverse capability, such that `InversionEnabler`
    actually has to run a conjugate gradient."""
    dom = ift.RGSpace(8, harmonic=True)
    ht = ift.HarmonicTransformOperator(dom)
    diag = ift.from_random(ht.target, device_id=device_id).exp()
    return ift.ScalingOperator(dom, 1.) + ht.adjoint @ ift.DiagonalOperator(diag) @ ht


def _multi_sum_operator(device_id):
    dom = ift.makeDomain({"a": ift.RGSpace(8), "b": ift.RGSpace((3, 4))})
    diag = ift.from_random(dom, device_id=device_id).exp()
    return ift.ScalingOperator(dom, 1.) + ift.makeOp(diag)


def test_inversion_enabler_device(device_id):
    for op in [_sum_operator(device_id), _multi_sum_operator(device_id)]:
        assert not op.capability & op.INVERSE_TIMES
        iop = ift.InversionEnabler(op, _ic)
        x = ift.from_random(iop.target, device_id=device_id)
        assert iop.inverse_times(x).device_id == x.device_id
        assert iop.inverse(x).device_id == x.device_id


def test_inversion_enabler_consistency():
    for op in [_sum_operator(-1), _multi_sum_operator(-1)]:
        iop = ift.InversionEnabler(op, _ic)
        ift.extra.check_linear_operator(iop, np.float64, np.float64,
                                        atol=1e-7, rtol=1e-7)
