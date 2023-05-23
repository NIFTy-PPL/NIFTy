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
# Copyright(C) 2013-2023 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import nifty8 as ift


def test_linear_wpprior():
    sp1 = ift.RGSpace(10, distances=1)
    sp2 = ift.RGSpace([3, 3])
    dom = ift.makeDomain([sp2, sp1])

    amp = ift.Field.from_raw(dom, 1)
    wp = ift.WPPrior(amp, space=1)
    xi = ift.from_random(wp.domain)
    res = wp(xi)

    x1 = res.val
    x2 = xi["xi"].val

    diff = np.zeros(x1.shape)
    diff[:, :, 1:] = np.diff(x1, axis=2)
    diff[:, :, 0] = x1[:, :, 0]
    np.testing.assert_allclose(diff, x2)

    # do linear operator checks
    ift.extra.check_linear_operator(wp)


def test_non_linear_wpprior():
    sp1 = ift.RGSpace(10, distances=1)
    sp2 = ift.RGSpace([3, 3])
    dom = ift.makeDomain([sp2, sp1])

    amp = ift.ScalingOperator(dom, 1).ducktape("amplitude")
    wp = ift.WPPrior(amp, space=1)

    one_field = ift.MultiField.full(amp.domain, 1)
    xi = ift.from_random(wp.domain)
    xi_amp_one = ift.MultiField.union((xi, one_field))
    res = wp(xi_amp_one)

    x1 = res.val
    x2 = xi["xi"].val
    diff = np.zeros(x1.shape)
    diff[:, :, 0] = x1[:, :, 0]
    diff[:, :, 1:] = np.diff(x1, axis=2)
    np.testing.assert_allclose(diff, x2)

    # do the operator checks for non-linear ops
    ift.extra.check_operator(wp, xi)
