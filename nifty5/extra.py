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

from .field import Field
from .linearization import Linearization
from .sugar import from_random

__all__ = ["consistency_check", "check_value_gradient_consistency",
           "check_value_gradient_metric_consistency"]


def _assert_allclose(f1, f2, atol, rtol):
    if isinstance(f1, Field):
        return np.testing.assert_allclose(f1.local_data, f2.local_data,
                                          atol=atol, rtol=rtol)
    for key, val in f1.items():
        _assert_allclose(val, f2[key], atol=atol, rtol=rtol)


def _adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol):
    needed_cap = op.TIMES | op.ADJOINT_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    f1 = from_random("normal", op.domain, dtype=domain_dtype)
    f2 = from_random("normal", op.target, dtype=target_dtype)
    res1 = f1.vdot(op.adjoint_times(f2))
    res2 = op.times(f1).vdot(f2)
    np.testing.assert_allclose(res1, res2, atol=atol, rtol=rtol)


def _inverse_implementation(op, domain_dtype, target_dtype, atol, rtol):
    needed_cap = op.TIMES | op.INVERSE_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    foo = from_random("normal", op.target, dtype=target_dtype)
    res = op(op.inverse_times(foo))
    _assert_allclose(res, foo, atol=atol, rtol=rtol)

    foo = from_random("normal", op.domain, dtype=domain_dtype)
    res = op.inverse_times(op(foo))
    _assert_allclose(res, foo, atol=atol, rtol=rtol)


def _full_implementation(op, domain_dtype, target_dtype, atol, rtol):
    _adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol)
    _inverse_implementation(op, domain_dtype, target_dtype, atol, rtol)


def consistency_check(op, domain_dtype=np.float64, target_dtype=np.float64,
                      atol=0, rtol=1e-7):
    _full_implementation(op, domain_dtype, target_dtype, atol, rtol)
    _full_implementation(op.adjoint, target_dtype, domain_dtype, atol, rtol)
    _full_implementation(op.inverse, target_dtype, domain_dtype, atol, rtol)
    _full_implementation(op.adjoint.inverse, domain_dtype, target_dtype, atol,
                         rtol)


def _get_acceptable_location(op, loc, lin):
    if not np.isfinite(lin.val.sum()):
        raise ValueError('Initial value must be finite')
    dir = from_random("normal", loc.domain)
    dirder = lin.jac(dir)
    if dirder.norm() == 0:
        dir = dir * (lin.val.norm()*1e-5)
    else:
        dir = dir * (lin.val.norm()*1e-5/dirder.norm())
    # Find a step length that leads to a "reasonable" location
    for i in range(50):
        try:
            loc2 = loc+dir
            lin2 = op(Linearization.make_var(loc2, lin.want_metric))
            if np.isfinite(lin2.val.sum()) and abs(lin2.val.sum()) < 1e20:
                break
        except FloatingPointError:
            pass
        dir = dir*0.5
    else:
        raise ValueError("could not find a reasonable initial step")
    return loc2, lin2


def _check_consistency(op, loc, tol, ntries, do_metric):
    for _ in range(ntries):
        lin = op(Linearization.make_var(loc, do_metric))
        loc2, lin2 = _get_acceptable_location(op, loc, lin)
        dir = loc2-loc
        locnext = loc2
        dirnorm = dir.norm()
        for i in range(50):
            locmid = loc + 0.5*dir
            linmid = op(Linearization.make_var(locmid, do_metric))
            dirder = linmid.jac(dir)
            numgrad = (lin2.val-lin.val)
            xtol = tol * dirder.norm() / np.sqrt(dirder.size)
            cond = (abs(numgrad-dirder) <= xtol).all()
            if do_metric:
                dgrad = linmid.metric(dir)
                dgrad2 = (lin2.gradient-lin.gradient)
                cond = cond and (abs(dgrad-dgrad2) <= xtol).all()
            if cond:
                break
            dir = dir*0.5
            dirnorm *= 0.5
            loc2, lin2 = locmid, linmid
        else:
            raise ValueError("gradient and value seem inconsistent")
        loc = locnext


def check_value_gradient_consistency(op, loc, tol=1e-8, ntries=100):
    _check_consistency(op, loc, tol, ntries, False)


def check_value_gradient_metric_consistency(op, loc, tol=1e-8, ntries=100):
    _check_consistency(op, loc, tol, ntries, True)
