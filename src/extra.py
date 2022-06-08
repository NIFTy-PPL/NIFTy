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
# Copyright(C) 2013-2022 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from itertools import combinations

import numpy as np

from .domain_tuple import DomainTuple
from .field import Field
from .linearization import Linearization
from .multi_domain import MultiDomain
from .multi_field import MultiField
from .operators.adder import Adder
from .operators.endomorphic_operator import EndomorphicOperator
from .operators.energy_operators import (EnergyOperator,
                                         LikelihoodEnergyOperator)
from .operators.linear_operator import LinearOperator
from .operators.operator import Operator
from .probing import StatCalculator
from .sugar import from_random, is_fieldlike, is_operator
from .utilities import issingleprec, myassert

__all__ = ["check_linear_operator", "check_operator", "assert_allclose", "minisanity"]


def check_linear_operator(op, domain_dtype=np.float64, target_dtype=np.float64,
                          atol=1e-12, rtol=1e-12, only_r_linear=False):
    """Checks an operator for algebraic consistency of its capabilities.

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
    _purity_check(op, from_random(op.domain, dtype=domain_dtype))
    _purity_check(op.adjoint.inverse, from_random(op.domain, dtype=domain_dtype))
    _purity_check(op.adjoint, from_random(op.target, dtype=target_dtype))
    _purity_check(op.inverse, from_random(op.target, dtype=target_dtype))
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
    _check_sqrt(op, domain_dtype)
    _check_sqrt(op.adjoint, target_dtype)
    _check_sqrt(op.inverse, target_dtype)
    _check_sqrt(op.adjoint.inverse, domain_dtype)


def check_operator(op, loc, tol=1e-12, ntries=100, perf_check=True,
                   only_r_differentiable=True, metric_sampling=True):
    """Performs various checks of the implementation of linear and nonlinear
    operators.

    Computes the Jacobian with finite differences and compares it to the
    implemented Jacobian.

    Parameters
    ----------
    op : Operator
        Operator which shall be checked.
    loc : :class:`nifty8.field.Field` or :class:`nifty8.multi_field.MultiField`
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
        raise TypeError('This test tests only (nonlinear) operators.')
    _domain_check_nonlinear(op, loc)
    _purity_check(op, loc)
    _performance_check(op, loc, bool(perf_check))
    _linearization_value_consistency(op, loc)
    _jac_vs_finite_differences(op, loc, np.sqrt(tol), ntries,
                               only_r_differentiable)
    _check_nontrivial_constant(op, loc, tol, ntries, only_r_differentiable,
                               metric_sampling)
    _check_likelihood_energy(op, loc)


def assert_allclose(f1, f2, atol=0, rtol=1e-7):
    if isinstance(f1, Field):
        return np.testing.assert_allclose(f1.val, f2.val, atol=atol, rtol=rtol)
    if f1.domain is not f2.domain:
        raise AssertionError
    for key, val in f1.items():
        assert_allclose(val, f2[key], atol=atol, rtol=rtol)


def assert_equal(f1, f2):
    if isinstance(f1, Field):
        return np.testing.assert_equal(f1.val, f2.val)
    if f1.domain is not f2.domain:
        raise AssertionError
    for key, val in f1.items():
        assert_equal(val, f2[key])


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
    myassert(inp.domain is op.domain)
    myassert(op(inp).domain is op.target)


def _check_sqrt(op, domain_dtype):
    if not isinstance(op, EndomorphicOperator):
        try:
            op.get_sqrt()
            raise RuntimeError("Operator implements get_sqrt() although it is not an endomorphic operator.")
        except AttributeError:
            return
    try:
        sqop = op.get_sqrt()
    except (NotImplementedError, ValueError):
        return
    fld = from_random(op.domain, dtype=domain_dtype)
    a = op(fld)
    b = (sqop.adjoint @ sqop)(fld)
    return assert_allclose(a, b, rtol=1e-15)


def _domain_check_nonlinear(op, loc):
    _domain_check(op)
    myassert(isinstance(loc, (Field, MultiField)))
    myassert(loc.domain is op.domain)
    for wm in [False, True]:
        lin = Linearization.make_var(loc, wm)
        reslin = op(lin)
        myassert(lin.domain is op.domain)
        myassert(lin.target is op.domain)
        myassert(lin.val.domain is lin.domain)
        myassert(reslin.domain is op.domain)
        myassert(reslin.target is op.target)
        myassert(reslin.val.domain is reslin.target)
        myassert(reslin.target is op.target)
        myassert(reslin.jac.domain is reslin.domain)
        myassert(reslin.jac.target is reslin.target)
        myassert(lin.want_metric == reslin.want_metric)
        _domain_check_linear(reslin.jac, inp=loc)
        _domain_check_linear(reslin.jac.adjoint, inp=reslin.jac(loc))
        if reslin.metric is not None:
            myassert(reslin.metric.domain is reslin.metric.target)
            myassert(reslin.metric.domain is op.domain)


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
        lin = myop(Linearization.make_var(pos, wm))
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


def _purity_check(op, pos):
    if isinstance(op, LinearOperator) and (op.capability & op.TIMES) != op.TIMES:
        return
    res0 = op(pos)
    res1 = op(pos)
    assert_equal(res0, res1)


def _get_acceptable_location(op, loc, lin):
    if not np.isfinite(lin.val.s_sum()):
        raise ValueError('Initial value must be finite')
    direction = from_random(loc.domain, dtype=loc.dtype)
    dirder = lin.jac(direction)
    fac = 1e-3 if issingleprec(loc.dtype) else 1e-6
    if dirder.norm() == 0:
        direction = direction * (lin.val.norm() * fac)
    else:
        direction = direction * (lin.val.norm() * fac / dirder.norm())
    # Find a step length that leads to a "reasonable" location
    for i in range(50):
        try:
            loc2 = loc + direction
            lin2 = op(Linearization.make_var(loc2, lin.want_metric))
            if np.isfinite(lin2.val.s_sum()) and abs(lin2.val.s_sum()) < 1e20:
                break
        except FloatingPointError:
            pass
        direction = direction * 0.5
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
    if isinstance(op.domain, DomainTuple):
        return
    keys = op.domain.keys()
    combis = []
    if len(keys) > 4:
        from .logger import logger
        logger.warning('Operator domain has more than 4 keys.')
        logger.warning('Check derivatives only with one constant key at a time.')
        combis = [[kk] for kk in keys]
    else:
        for ll in range(1, len(keys)):
            combis.extend(list(combinations(keys, ll)))
    for cstkeys in combis:
        varkeys = set(keys) - set(cstkeys)
        cstloc = loc.extract_by_keys(cstkeys)
        varloc = loc.extract_by_keys(varkeys)

        val0 = op(loc)
        _, op0 = op.simplify_for_constant_input(cstloc)
        myassert(op0.domain is varloc.domain)
        val1 = op0(varloc)
        assert_equal(val0, val1)

        lin = Linearization.make_partial_var(loc, cstkeys, want_metric=True)
        lin0 = Linearization.make_var(varloc, want_metric=True)
        oplin0 = op0(lin0)
        oplin = op(lin)

        myassert(oplin.jac.target is oplin0.jac.target)
        rndinp = from_random(oplin.jac.target, dtype=oplin.val.dtype)
        assert_allclose(oplin.jac.adjoint(rndinp).extract(varloc.domain),
                        oplin0.jac.adjoint(rndinp), 1e-13, 1e-13)
        foo = oplin.jac.adjoint(rndinp).extract(cstloc.domain)
        assert_equal(foo, 0*foo)

        if isinstance(op, EnergyOperator) and metric_sampling:
            oplin.metric.draw_sample()

        # _jac_vs_finite_differences(op0, varloc, np.sqrt(tol), ntries,
        #                            only_r_differentiable)


def _jac_vs_finite_differences(op, loc, tol, ntries, only_r_differentiable):
    for _ in range(ntries):
        lin = op(Linearization.make_var(loc))
        loc2, lin2 = _get_acceptable_location(op, loc, lin)
        direction = loc2 - loc
        locnext = loc2
        dirnorm = direction.norm()
        hist = []
        for i in range(50):
            locmid = loc + 0.5 * direction
            linmid = op(Linearization.make_var(locmid))
            dirder = linmid.jac(direction)
            numgrad = (lin2.val - lin.val)
            xtol = tol * dirder.norm() / np.sqrt(dirder.size)
            hist.append((numgrad - dirder).norm())
            # print(len(hist),hist[-1])
            if (abs(numgrad - dirder) <= xtol).s_all():
                break
            direction = direction * 0.5
            dirnorm *= 0.5
            loc2, lin2 = locmid, linmid
        else:
            print(hist)
            raise ValueError("gradient and value seem inconsistent")
        loc = locnext
        check_linear_operator(linmid.jac, domain_dtype=loc.dtype,
                              target_dtype=dirder.dtype,
                              only_r_linear=only_r_differentiable,
                              atol=tol**2, rtol=tol**2)


def _check_likelihood_energy(op, loc):
    from .operators.energy_operators import LikelihoodEnergyOperator
    if not isinstance(op, LikelihoodEnergyOperator):
        return
    data_domain = op.data_domain
    if data_domain is None:
        return
    smet = op._sqrt_data_metric_at(loc)
    myassert(smet.domain == smet.target == data_domain)
    nres = op.normalized_residual(loc)
    myassert(nres.domain is data_domain)
    res = op.get_transformation()
    if res is None:
        raise RuntimeError("`get_transformation` is not implemented for "
                            "this LikelihoodEnergyOperator")
    if len(res) != 2:
        raise RuntimeError("`get_transformation` has to return a dtype and the transformation")


def minisanity(likelihood_energy, samples, terminal_colors=True, return_values=False):
    """Log information about the current fit quality and prior compatibility.

    Log a table with fitting information for the likelihood and the prior.
    Assume that the variables in `energy.position.domain` are standard-normal
    distributed a priori. The table contains the reduced chi^2 value, the mean
    and the number of degrees of freedom for every key of a `MultiDomain`. If
    the domain is a `DomainTuple`, the displayed key is `<None>`.

    If everything is consistent the reduced chi^2 values should be close to one
    and the mean of the data residuals close to zero. If the reduced chi^2 value
    in latent space is significantly bigger than one and only one degree of
    freedom is present, the mean column gives an indication in which direction
    to change the respective hyper parameters.

    Ignore all NaN entries in the target of `modeldata_operator` and in `data`.
    Print reduced chi-square values above 2 and 5 in orange and red,
    respectively.

    Parameters
    ----------
    likelihood_energy: LikelihoodEnergyOperator
        Likelihood energy of which the normalized residuals shall be computed.

    samples : SampleListBase
        List of samples.

    terminal_colors : bool, optional
        Setting this to false disables terminal colors. This may be useful if
        the output of minisanity is written to a file. Default: True

    return_values : bool, optional
        If true, in addition to the table in string format, `minisanity` will
        return the computed values as a dictionary. Default: `False`.

    Returns
    -------
        printable_table : string
        values : dictionary
            Only returned if `return_values` is `True`

    Note
    ----
    For computing the reduced chi^2 values and the normalized residuals, the
    metric at `mean` is used.

    """
    from .minimization.sample_list import SampleListBase
    from .sugar import makeDomain

    if not isinstance(samples, SampleListBase):
        raise TypeError(
            "Minisanity takes only SampleLists as input. If you happen to have "
            "only one field (i.e. no samples), you may wrap it via "
            "`ift.SampleList([field])` and pass it to minisanity."
        )

    if not isinstance(likelihood_energy, LikelihoodEnergyOperator):
        return ""

    data_domain = likelihood_energy.data_domain
    latent_domain = samples.domain
    xdoms = [data_domain, latent_domain]

    keylen = 18
    for dom in xdoms:
        if isinstance(dom, MultiDomain):
            keylen = max([max(map(len, dom.keys())), keylen])
    keylen = min([keylen, 42])

    # compute xops
    xops = []
    nres = likelihood_energy.normalized_residual
    if isinstance(data_domain, MultiDomain):
        lam = lambda x: nres(x)
    else:
        name = likelihood_energy.name
        if name is None:
            name = "<None>"
        data_domain = makeDomain({name: data_domain})
        lam = lambda x: nres(x).ducktape_left(name)
    xops.append(lam)
    if isinstance(latent_domain, MultiDomain):
        xops.append(lambda x: x)
    else:
        latent_domain = makeDomain({"<None>": latent_domain})
        xops.append(lambda x: x.ducktape_left("<None>"))
    # /compute xops

    xdoms = [data_domain, latent_domain]
    xredchisq, xscmean, xndof = [], [], []
    for dd in xdoms:
        xredchisq.append({kk: StatCalculator() for kk in dd.keys()})
        xscmean.append({kk: StatCalculator() for kk in dd.keys()})
        xndof.append({})

    for ss1, ss2 in zip(samples.iterator(xops[0]), samples.iterator(xops[1])):
        if isinstance(data_domain, MultiDomain):
            myassert(ss1.domain == data_domain)
        if isinstance(samples.domain, MultiDomain):
            myassert(ss2.domain == samples.domain)
        for ii, ss in enumerate((ss1, ss2)):
            for kk in ss.domain.keys():
                lsize = ss[kk].size - np.sum(np.isnan(ss[kk].val))
                xredchisq[ii][kk].add(np.nansum(abs(ss[kk].val) ** 2) / lsize)
                xscmean[ii][kk].add(np.nanmean(ss[kk].val))
                xndof[ii][kk] = lsize

    for ii in range(2):
        for kk in xredchisq[ii].keys():
            rcs_mean = xredchisq[ii][kk].mean
            sc_mean = xscmean[ii][kk].mean
            try:
                rcs_std = np.sqrt(xredchisq[ii][kk].var)
                sc_std = np.sqrt(xscmean[ii][kk].var)
            except RuntimeError:
                rcs_std = None
                sc_std = None
            xredchisq[ii][kk] = {'mean': rcs_mean, 'std': rcs_std}
            xscmean[ii][kk] = {'mean': sc_mean, 'std': sc_std}

    s0 = _tableentries(xredchisq[0], xscmean[0], xndof[0], keylen, terminal_colors)
    s1 = _tableentries(xredchisq[1], xscmean[1], xndof[1], keylen, terminal_colors)

    n = 38 + keylen
    s = [n * "=",
         (keylen + 2) * " " + "{:>11}".format("reduced χ²")
           + "{:>14}".format("mean") + "{:>11}".format("# dof"),
         n * "-",
         "Data residuals", s0,
         "Latent space", s1,
         n * "="]

    res_string = "\n".join(s)

    if not return_values:
        return res_string
    else:
        res_dict = {
            'redchisq': {
                'data_residuals': xredchisq[0],
                'latent_variables': xredchisq[1]
            },
            'scmean': {
                'data_residuals': xscmean[0],
                'latent_variables': xscmean[1],
            },
            'ndof': {
                'data_residuals': xndof[0],
                'latent_variables': xndof[1]
            }
        }
        return res_string, res_dict


def _tableentries(redchisq, scmean, ndof, keylen, colors):
    class _bcolors:
        WARNING = "\033[33m" if colors else ""
        FAIL = "\033[31m" if colors else ""
        ENDC = "\033[0m" if colors else ""
        BOLD = "\033[1m" if colors else ""

    out = ""
    for kk in redchisq.keys():
        if len(kk) > keylen:
            out += "  " + kk[: keylen - 1] + "…"
        else:
            out += "  " + kk.ljust(keylen)
        foo = f"{redchisq[kk]['mean']:.1f}"
        if redchisq[kk]['std'] is not None:
            foo += f" ± {redchisq[kk]['std']:.1f}"
        if redchisq[kk]['mean'] > 5 or redchisq[kk]['mean'] < 1/5:
            out += _bcolors.FAIL + _bcolors.BOLD + f"{foo:>11}" + _bcolors.ENDC
        elif redchisq[kk]['mean'] > 2 or redchisq[kk]['mean'] < 1/2:
            out += _bcolors.WARNING + _bcolors.BOLD + f"{foo:>11}" + _bcolors.ENDC
        else:
            out += f"{foo:>11}"

        foo = f"{scmean[kk]['mean']:.1f}"
        if scmean[kk]['std'] is not None:
            foo += f" ± {scmean[kk]['std']:.1f}"
        out += f"{foo:>14}"
        out += f"{ndof[kk]:>11}"
        out += "\n"
    return out[:-1]
