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

import numpy as np
from ..field import Field
from .. import dobj

__all__ = ["consistency_check"]


def adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol):
    needed_cap = op.TIMES | op.ADJOINT_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    f1 = Field.from_random("normal", op.domain, dtype=domain_dtype)
    f2 = Field.from_random("normal", op.target, dtype=target_dtype)
    res1 = f1.vdot(op.adjoint_times(f2))
    res2 = op.times(f1).vdot(f2)
    np.testing.assert_allclose(res1, res2, atol=atol, rtol=rtol)


def inverse_implementation(op, domain_dtype, target_dtype, atol, rtol):
    needed_cap = op.TIMES | op.INVERSE_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    foo = Field.from_random("normal", op.target, dtype=target_dtype)
    res = op(op.inverse_times(foo))
    np.testing.assert_allclose(dobj.to_global_data(res.val),
                               dobj.to_global_data(foo.val),
                               atol=atol, rtol=rtol)

    foo = Field.from_random("normal", op.domain, dtype=domain_dtype)
    res = op.inverse_times(op(foo))
    np.testing.assert_allclose(dobj.to_global_data(res.val),
                               dobj.to_global_data(foo.val),
                               atol=atol, rtol=rtol)


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
