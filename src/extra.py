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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from itertools import combinations

import numpy as np
from numpy.testing import assert_

from .domain_tuple import DomainTuple
from .field import Field
from .linearization import Linearization
from .multi_domain import MultiDomain
from .multi_field import MultiField
from .operators.energy_operators import EnergyOperator
from .operators.linear_operator import LinearOperator
from .operators.operator import Operator
from .sugar import from_random, full, makeDomain

__all__ = ["check_linear_operator", "check_operator",
           "assert_allclose"]


def check_linear_operator(op, domain_dtype=np.float64, target_dtype=np.float64,
                          atol=0, rtol=1e-7, only_r_linear=False):
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
    only_r_linear: bool
        set to True if the operator is only R-linear, not C-linear.
        This will relax the adjointness test accordingly.
    """
    if not isinstance(op, LinearOperator):
        raise TypeError('This test tests only linear operators.')
    _domain_check_linear(op, domain_dtype)
    _domain_check_linear(op.adjoint, target_dtype)
    _domain_check_linear(op.inverse, target_dtype)
    _domain_check_linear(op.adjoint.inverse, domain_dtype)
    _check_linearity(op, domain_dtype, atol, rtol)
    _check_linearity(op.adjoint, target_dtype, atol, rtol)
    _check_linearity(op.inverse, target_dtype, atol, rtol)
    _check_linearity(op.adjoint.inverse, domain_dtype, atol, rtol)
    _full_implementation(op, domain_dtype, target_dtype, atol, rtol,
                         only_r_linear)
    _full_implementation(op.adjoint, target_dtype, domain_dtype, atol, rtol,
                         only_r_linear)
    _full_implementation(op.inverse, target_dtype, domain_dtype, atol, rtol,
                         only_r_linear)
    _full_implementation(op.adjoint.inverse, domain_dtype, target_dtype, atol,
                         rtol, only_r_linear)


def check_operator(op, loc, tol=1e-8, ntries=100, perf_check=True,
                   only_r_differentiable=True, metric_sampling=True):
    """
    Performs various checks of the implementation of linear and nonlinear
    operators.

    Computes the Jacobian with finite differences and compares it to the
    implemented Jacobian.

    Parameters
    ----------
    op : Operator
        Operator which shall be checked.
    loc : Field or MultiField
        An Field or MultiField instance which has the same domain
        as op. The location at which the gradient is checked
    tol : float
        Tolerance for the check.
    perf_check : Boolean
        Do performance check. May be disabled for very unimportant operators.
    only_r_differentiable : Boolean
        Jacobians of C-differentiable operators need to be C-linear.
        Default: True
    metric_sampling: Boolean
        If op is an EnergyOperator, metric_sampling determines whether the
        test shall try to sample from the metric or not.
    """
    if not isinstance(op, Operator):
        raise TypeError('This test tests only linear operators.')
    _domain_check_nonlinear(op, loc)
    _performance_check(op, loc, bool(perf_check))
    _linearization_value_consistency(op, loc)
    _jac_vs_finite_differences(op, loc, tol, ntries, only_r_differentiable)
    _check_nontrivial_constant(op, loc, tol, ntries, only_r_differentiable,
                               metric_sampling)


def assert_allclose(f1, f2, atol, rtol):
    if isinstance(f1, Field):
        return np.testing.assert_allclose(f1.val, f2.val, atol=atol, rtol=rtol)
    for key, val in f1.items():
        assert_allclose(val, f2[key], atol=atol, rtol=rtol)


def assert_equal(f1, f2):
    if isinstance(f1, Field):
        return np.testing.assert_equal(f1.val, f2.val)
    for key, val in f1.items():
        assert_equal(val, f2[key])


def _nozero(fld):
    if isinstance(fld, Field):
        return np.testing.assert_((fld != 0).s_all())
    for val in fld.values():
        _nozero(val)


def _allzero(fld):
    if isinstance(fld, Field):
        return np.testing.assert_((fld == 0.).s_all())
    for val in fld.values():
        _allzero(val)


def _adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol,
                            only_r_linear):
    needed_cap = op.TIMES | op.ADJOINT_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    f1 = from_random(op.domain, "normal", dtype=domain_dtype)
    f2 = from_random(op.target, "normal", dtype=target_dtype)
    res1 = f1.s_vdot(op.adjoint_times(f2))
    res2 = op.times(f1).s_vdot(f2)
    if only_r_linear:
        res1, res2 = res1.real, res2.real
    np.testing.assert_allclose(res1, res2, atol=atol, rtol=rtol)


def _inverse_implementation(op, domain_dtype, target_dtype, atol, rtol):
    needed_cap = op.TIMES | op.INVERSE_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    foo = from_random(op.target, "normal", dtype=target_dtype)
    res = op(op.inverse_times(foo))
    assert_allclose(res, foo, atol=atol, rtol=rtol)

    foo = from_random(op.domain, "normal", dtype=domain_dtype)
    res = op.inverse_times(op(foo))
    assert_allclose(res, foo, atol=atol, rtol=rtol)


def _full_implementation(op, domain_dtype, target_dtype, atol, rtol,
                         only_r_linear):
    _adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol,
                            only_r_linear)
    _inverse_implementation(op, domain_dtype, target_dtype, atol, rtol)


def _check_linearity(op, domain_dtype, atol, rtol):
    needed_cap = op.TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    fld1 = from_random(op.domain, "normal", dtype=domain_dtype)
    fld2 = from_random(op.domain, "normal", dtype=domain_dtype)
    alpha = 0.42
    val1 = op(alpha*fld1+fld2)
    val2 = alpha*op(fld1)+op(fld2)
    assert_allclose(val1, val2, atol=atol, rtol=rtol)


def _domain_check_linear(op, domain_dtype=None, inp=None):
    _domain_check(op)
    needed_cap = op.TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    if domain_dtype is not None:
        inp = from_random(op.domain, "normal", dtype=domain_dtype)
    elif inp is None:
        raise ValueError('Need to specify either dtype or inp')
    assert_(inp.domain is op.domain)
    assert_(op(inp).domain is op.target)


def _domain_check_nonlinear(op, loc):
    _domain_check(op)
    assert_(isinstance(loc, (Field, MultiField)))
    assert_(loc.domain is op.domain)
    for wm in [False, True]:
        lin = Linearization.make_var(loc, wm)
        reslin = op(lin)
        assert_(lin.domain is op.domain)
        assert_(lin.target is op.domain)
        assert_(lin.val.domain is lin.domain)
        assert_(reslin.domain is op.domain)
        assert_(reslin.target is op.target)
        assert_(reslin.val.domain is reslin.target)
        assert_(reslin.target is op.target)
        assert_(reslin.jac.domain is reslin.domain)
        assert_(reslin.jac.target is reslin.target)
        assert_(lin.want_metric == reslin.want_metric)
        _domain_check_linear(reslin.jac, inp=loc)
        _domain_check_linear(reslin.jac.adjoint, inp=reslin.jac(loc))
        if reslin.metric is not None:
            assert_(reslin.metric.domain is reslin.metric.target)
            assert_(reslin.metric.domain is op.domain)


def _domain_check(op):
    for dd in [op.domain, op.target]:
        if not isinstance(dd, (DomainTuple, MultiDomain)):
            raise TypeError(
                'The domain and the target of an operator need to',
                'be instances of either DomainTuple or MultiDomain.')


def _performance_check(op, pos, raise_on_fail):
    class CountingOp(LinearOperator):
        def __init__(self, domain):
            from .sugar import makeDomain
            self._domain = self._target = makeDomain(domain)
            self._capability = self.TIMES | self.ADJOINT_TIMES
            self._count = 0

        def apply(self, x, mode):
            self._count += 1
            return x

        @property
        def count(self):
            return self._count
    for wm in [False, True]:
        cop = CountingOp(op.domain)
        myop = op @ cop
        myop(pos)
        cond = [cop.count != 1]
        lin = myop(2*Linearization.make_var(pos, wm))
        cond.append(cop.count != 2)
        lin.jac(pos)
        cond.append(cop.count != 3)
        lin.jac.adjoint(lin.val)
        cond.append(cop.count != 4)
        if lin.metric is not None:
            lin.metric(pos)
            cond.append(cop.count != 6)
        if any(cond):
            s = 'The operator has a performance problem (want_metric={}).'.format(wm)
            from .logger import logger
            logger.error(s)
            logger.info(cond)
            if raise_on_fail:
                raise RuntimeError(s)


def _get_acceptable_location(op, loc, lin):
    if not np.isfinite(lin.val.s_sum()):
        raise ValueError('Initial value must be finite')
    dir = from_random(loc.domain, dtype=loc.dtype)
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
            if np.isfinite(lin2.val.s_sum()) and abs(lin2.val.s_sum()) < 1e20:
                break
        except FloatingPointError:
            pass
        dir = dir*0.5
    else:
        raise ValueError("could not find a reasonable initial step")
    return loc2, lin2


def _linearization_value_consistency(op, loc):
    for wm in [False, True]:
        lin = Linearization.make_var(loc, wm)
        fld0 = op(loc)
        fld1 = op(lin).val
        assert_allclose(fld0, fld1, 0, 1e-7)


def _check_nontrivial_constant(op, loc, tol, ntries, only_r_differentiable,
                               metric_sampling):
    return  # FIXME
    # Assumes that the operator is not constant
    if isinstance(op.domain, DomainTuple):
        return
    keys = op.domain.keys()
    for ll in range(0, len(keys)):
        for cstkeys in combinations(keys, ll):
            cstdom, vardom = {}, {}
            for kk, dd in op.domain.items():
                if kk in cstkeys:
                    cstdom[kk] = dd
                else:
                    vardom[kk] = dd
            cstdom, vardom = makeDomain(cstdom), makeDomain(vardom)
            cstloc = loc.extract(cstdom)

            val0 = op(loc)
            _, op0 = op.simplify_for_constant_input(cstloc)
            val1 = op0(loc)
            val2 = op0(loc.unite(cstloc))
            assert_equal(val1, val2)
            assert_equal(val0, val1)

            lin = Linearization.make_var(loc, want_metric=True)
            oplin = op0(lin)
            if isinstance(op, EnergyOperator):
                _allzero(oplin.gradient.extract(cstdom))
            _allzero(oplin.jac(from_random(cstdom).unite(full(vardom, 0))))

            if isinstance(op, EnergyOperator) and metric_sampling:
                samp0 = oplin.metric.draw_sample()
                _allzero(samp0.extract(cstdom))
                _nozero(samp0.extract(vardom))

            _jac_vs_finite_differences(op0, loc, tol, ntries, only_r_differentiable)


def _jac_vs_finite_differences(op, loc, tol, ntries, only_r_differentiable):
    for _ in range(ntries):
        lin = op(Linearization.make_var(loc))
        loc2, lin2 = _get_acceptable_location(op, loc, lin)
        dir = loc2-loc
        locnext = loc2
        dirnorm = dir.norm()
        hist = []
        for i in range(50):
            locmid = loc + 0.5*dir
            linmid = op(Linearization.make_var(locmid))
            dirder = linmid.jac(dir)
            numgrad = (lin2.val-lin.val)
            xtol = tol * dirder.norm() / np.sqrt(dirder.size)
            hist.append((numgrad-dirder).norm())
#            print(len(hist),hist[-1])
            if (abs(numgrad-dirder) <= xtol).s_all():
                break
            dir = dir*0.5
            dirnorm *= 0.5
            loc2, lin2 = locmid, linmid
        else:
            print(hist)
            raise ValueError("gradient and value seem inconsistent")
        loc = locnext
        check_linear_operator(linmid.jac, domain_dtype=loc.dtype,
                              target_dtype=dirder.dtype,
                              only_r_linear=only_r_differentiable)
