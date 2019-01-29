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
from .operators.linear_operator import LinearOperator
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


def _check_linearity(op, domain_dtype, atol, rtol):
    fld1 = from_random("normal", op.domain, dtype=domain_dtype)
    fld2 = from_random("normal", op.domain, dtype=domain_dtype)
    alpha = np.random.random()
    val1 = op(alpha*fld1+fld2)
    val2 = alpha*op(fld1)+op(fld2)
    _assert_allclose(val1, val2, atol=atol, rtol=rtol)


def consistency_check(op, domain_dtype=np.float64, target_dtype=np.float64,
                      atol=0, rtol=1e-7):
    """
    Checks an operator for algebraic consistency of its capabilities.

    Checks whether times(), adjoint_times(), inverse_times() and
    adjoint_inverse_times() (if in capability list) is implemented
    consistently. Additionally, it checks whether the operator is linear.

    Parameters
    ----------
    op : LinearOperator
        Operator which shall be checked.
    domain_dtype : dtype
        The data type of the random vectors in the operator's domain. Default
        is `np.float64`.
    target_dtype : dtype
        The data type of the random vectors in the operator's target. Default
        is `np.float64`.
    atol : float
        Absolute tolerance for the check. If rtol is specified, 
        then satisfying any tolerance will let the check pass. 
        Default: 0.
    rtol : float
        Relative tolerance for the check. If atol is specified, 
        then satisfying any tolerance will let the check pass. 
        Default: 0.
    """
    if not isinstance(op, LinearOperator):
        raise TypeError('This test tests only linear operators.')
    _check_linearity(op, domain_dtype, atol, rtol)
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


def _check_consistency(op, loc, tol, ntries):
    for _ in range(ntries):
        lin = op(Linearization.make_var(loc))
        loc2, lin2 = _get_acceptable_location(op, loc, lin)
        dir = loc2-loc
        locnext = loc2
        dirnorm = dir.norm()
        for i in range(50):
            locmid = loc + 0.5*dir
            linmid = op(Linearization.make_var(locmid))
            dirder = linmid.jac(dir)
            numgrad = (lin2.val-lin.val)
            xtol = tol * dirder.norm() / np.sqrt(dirder.size)
            cond = (abs(numgrad-dirder) <= xtol).all()
            if cond:
                break
            dir = dir*0.5
            dirnorm *= 0.5
            loc2, lin2 = locmid, linmid
        else:
            raise ValueError("gradient and value seem inconsistent")
        loc = locnext


def check_value_gradient_consistency(op, loc, tol=1e-8, ntries=100):
    """
    Checks the gradient (jacobian) of an operator against its value. 

    Computes the gradient (jacobian) with finite differences and compares
    it to the implemented gradient (jacobian).

    Parameters
    ----------
    op : Operator
        Operator which shall be checked.
    loc : Field or MultiField
        An Field or MultiField instance which has the same domain
        as op. The location at which the gradient is checked
    atol : float
        Absolute tolerance for the check. If rtol is specified, 
        then satisfying any tolerance will let the check pass. 
        Default: 0.
    rtol : float
        Relative tolerance for the check. If atol is specified, 
        then satisfying any tolerance will let the check pass. 
        Default: 0
    """
    _check_consistency(op, loc, tol, ntries)

