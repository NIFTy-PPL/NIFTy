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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function
from ..compat import *
import numpy as np
from ..sugar import from_random
from ..field import Field

__all__ = ["consistency_check"]


def _assert_allclose(f1, f2, atol, rtol):
    if isinstance(f1, Field):
        return np.testing.assert_allclose(f1.local_data, f2.local_data,
                                          atol=atol, rtol=rtol)
    for key, val in f1.items():
        _assert_allclose(val, f2[key], atol=atol, rtol=rtol)


def adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol):
    needed_cap = op.TIMES | op.ADJOINT_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    f1 = from_random("normal", op.domain, dtype=domain_dtype)
    f2 = from_random("normal", op.target, dtype=target_dtype)
    res1 = f1.vdot(op.adjoint_times(f2))
    res2 = op.times(f1).vdot(f2)
    np.testing.assert_allclose(res1, res2, atol=atol, rtol=rtol)


def inverse_implementation(op, domain_dtype, target_dtype, atol, rtol):
    needed_cap = op.TIMES | op.INVERSE_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    foo = from_random("normal", op.target, dtype=target_dtype)
    res = op(op.inverse_times(foo))
    _assert_allclose(res, foo, atol=atol, rtol=rtol)

    foo = from_random("normal", op.domain, dtype=domain_dtype)
    res = op.inverse_times(op(foo))
    _assert_allclose(res, foo, atol=atol, rtol=rtol)


def full_implementation(op, domain_dtype, target_dtype, atol, rtol):
    adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol)
    inverse_implementation(op, domain_dtype, target_dtype, atol, rtol)


def consistency_check(op, domain_dtype=np.float64, target_dtype=np.float64,
                      atol=0, rtol=1e-7):
    full_implementation(op, domain_dtype, target_dtype, atol, rtol)
    full_implementation(op.adjoint, target_dtype, domain_dtype, atol, rtol)
    full_implementation(op.inverse, target_dtype, domain_dtype, atol, rtol)
    full_implementation(op.adjoint.inverse, domain_dtype, target_dtype, atol,
                        rtol)
