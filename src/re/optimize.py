# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from datetime import datetime
from functools import partial
from typing import (
    Any, Callable, Dict, Mapping, NamedTuple, Optional, Tuple, Union
)

from jax import lax
from jax import numpy as jnp
from jax.tree_util import Partial

from . import conjugate_gradient
from .logger import logger
from .misc import conditional_call, conditional_raise, doc_from
from .tree_math import PyTreeString, assert_arithmetics, hide_strings
from .tree_math import norm as jft_norm
from .tree_math import result_type, size, vdot, where


class OptimizeResults(NamedTuple):
    """Object holding optimization results inspired by JAX and scipy.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. NOTE, in contrast to scipy there is no `message` for
        details since strings are not statically memory bound.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    """
    x: Any
    success: Union[bool, jnp.ndarray]
    status: Union[int, jnp.ndarray]
    fun: Any
    jac: Any
    hess: Optional[jnp.ndarray] = None
    hess_inv: Optional[jnp.ndarray] = None
    nfev: Union[None, int, jnp.ndarray] = None
    njev: Union[None, int, jnp.ndarray] = None
    nhev: Union[None, int, jnp.ndarray] = None
    nit: Union[None, int, jnp.ndarray] = None
    # Trust-Region specific slots
    trust_radius: Union[None, float, jnp.ndarray] = None
    jac_magnitude: Union[None, float, jnp.ndarray] = None
    good_approximation: Union[None, bool, jnp.ndarray] = None


def _prepare_vag_hessp(fun, jac, hessp,
                       fun_and_grad) -> Tuple[Callable, Callable]:
    """Returns a tuple of functions for computing the value-and-gradient and
    the Hessian-Vector-Product.
    """
    from warnings import warn

    if fun_and_grad is None:
        if fun is not None and jac is not None:
            uw = "computing the function together with its gradient would be faster"
            warn(uw, UserWarning)

            def fun_and_grad(x):
                return (fun(x), jac(x))
        elif fun is not None:
            from jax import value_and_grad

            fun_and_grad = value_and_grad(fun)
        else:
            ValueError("no function specified")

    if hessp is None:
        from jax import grad, jvp

        jac = grad(fun) if jac is None else jac

        def hessp(primals, tangents):
            return jvp(jac, (primals, ), (tangents, ))[1]

    return fun_and_grad, hessp


def newton_cg(fun=None, x0=None, *args, **kwargs):
    """Minimize a scalar-valued function using the Newton-CG algorithm."""
    if x0 is not None:
        assert_arithmetics(x0)
    return _newton_cg(fun, x0, *args, **kwargs).x


@doc_from(newton_cg)
def static_newton_cg(fun=None, x0=None, *args, **kwargs):
    if x0 is not None:
        assert_arithmetics(x0)
    return _static_newton_cg(fun, x0, *args, **kwargs).x


def _ncg_pretty_print_it(
    name,
    i,
    *,
    energy,
    energy_diff,
    grad_scaling,
    ls_reset,
    nhev,
    descent_norm,
    xtol,
    absdelta=None
):
    msg = (
        f"{name}: →:{grad_scaling} ↺:{ls_reset} #∇²:{nhev:02d}"
        f" |↘|:{descent_norm:.6e} 🞋:{xtol:.6e}"
        f"\n{name}: Iteration {i} ⛰:{energy:+.6e} Δ⛰:{energy_diff:.6e}" +
        (f" 🞋:{absdelta:.6e}" if absdelta is not None else "")
    )
    logger.info(msg)


def _newton_cg(
    fun=None,
    x0=None,
    *,
    miniter=None,
    maxiter=None,
    energy_reduction_factor=0.1,
    old_fval=None,
    absdelta=None,
    norm_ord=None,
    xtol=1e-5,
    jac: Optional[Callable] = None,
    fun_and_grad=None,
    hessp=None,
    name=None,
    cg=conjugate_gradient._cg,
    cg_kwargs=None,
    time_threshold=None,
    custom_gradnorm=None,
):
    norm_ord = 1 if norm_ord is None else norm_ord
    miniter = 0 if miniter is None else miniter
    maxiter = 200 if maxiter is None else maxiter
    xtol = xtol * size(x0)

    pos = x0
    fun_and_grad, hessp = _prepare_vag_hessp(
        fun, jac, hessp, fun_and_grad=fun_and_grad
    )
    cg_kwargs = {} if cg_kwargs is None else cg_kwargs
    cg_name = name + "CG" if name is not None else None

    pp = partial(_ncg_pretty_print_it, name, xtol=xtol, absdelta=absdelta)

    gradnorm = (
        partial(jft_norm, ord=norm_ord)
        if custom_gradnorm is None else custom_gradnorm
    )
    energy, g = fun_and_grad(pos)
    nfev, njev, nhev = 1, 1, 0
    if jnp.isnan(energy):
        raise ValueError("energy is Nan")
    status = -1
    i = 0
    for i in range(1, maxiter + 1):
        # Newton approximates the potential up to second order. The CG energy
        # (`0.5 * x.T @ A @ x - x.T @ b`) and the approximation to the true
        # potential in Newton thus live on comparable energy scales. Hence, the
        # energy in a Newton minimization can be used to set the CG energy
        # convergence criterion.
        if old_fval and energy_reduction_factor:
            cg_absdelta = energy_reduction_factor * (old_fval - energy)
        else:
            cg_absdelta = None if absdelta is None else absdelta / 100.
        mag_g = jft_norm(g, ord=cg_kwargs.get("norm_ord", 1))
        cg_resnorm = jnp.minimum(
            0.5, jnp.sqrt(mag_g)
        ) * mag_g  # taken from SciPy
        default_kwargs = {
            "absdelta": cg_absdelta,
            "resnorm": cg_resnorm,
            "norm_ord": 1,
            "_raise_nonposdef": False,  # handle non-pos-def
            "name": cg_name,
            "time_threshold": time_threshold
        }
        cg_res = cg(Partial(hessp, pos), g, **{**default_kwargs, **cg_kwargs})
        nat_g, info = cg_res.x, cg_res.info
        nhev += cg_res.nfev
        if info is not None and info < 0:
            raise ValueError("conjugate gradient failed")

        naive_ls_it = 0
        dd = nat_g  # negative descent direction
        grad_scaling = 1.
        ls_reset = False
        for naive_ls_it in range(9):
            new_pos = pos - grad_scaling * dd
            new_energy, new_g = fun_and_grad(new_pos)
            nfev, njev = nfev + 1, njev + 1
            if new_energy <= energy:
                break

            grad_scaling /= 2
            if naive_ls_it == 5:
                ls_reset = True
                gam = float(vdot(g, g))
                curv = float(g.dot(hessp(pos, g)))
                nhev += 1
                grad_scaling = 1.
                dd = gam / curv * g
        else:
            grad_scaling = 0.
            nm = "N" if name is None else name
            msg = f"{nm}: WARNING: Energy would increase; aborting"
            logger.warning(msg)
            status = -1
            break

        energy_diff = energy - new_energy
        old_fval = energy
        energy = new_energy
        pos = new_pos
        g = new_g

        descent_norm = grad_scaling * gradnorm(dd)
        if name is not None:
            pp(
                i,
                energy=energy,
                energy_diff=energy_diff,
                grad_scaling=grad_scaling,
                ls_reset=ls_reset,
                nhev=nhev,
                descent_norm=descent_norm,
            )
        if jnp.isnan(new_energy):
            raise ValueError("energy is NaN")
        min_cond = naive_ls_it < 2 and i > miniter
        if absdelta is not None and 0. <= energy_diff < absdelta and min_cond:
            status = 0
            break
        if descent_norm <= xtol and i > miniter:
            status = 0
            break
        if time_threshold is not None and datetime.now() > time_threshold:
            status = i
            break
    else:
        status = i
        nm = "N" if name is None else name
        logger.error(f"{nm}: Iteration Limit Reached")
    return OptimizeResults(
        x=pos,
        success=True,
        status=status,
        fun=energy,
        jac=g,
        nit=i,
        nfev=nfev,
        njev=njev,
        nhev=nhev
    )


def _static_newton_cg(
    fun=None,
    x0=None,
    *,
    miniter=None,
    maxiter=None,
    energy_reduction_factor=0.1,
    old_fval=None,
    absdelta=None,
    norm_ord=None,
    xtol=1e-5,
    jac: Optional[Callable] = None,
    fun_and_grad=None,
    hessp=None,
    cg=conjugate_gradient._static_cg,
    name=None,
    cg_kwargs=None,
    custom_gradnorm=None,
):
    from jax.debug import callback

    from .lax import while_loop

    norm_ord = 1 if norm_ord is None else norm_ord
    miniter = 0 if miniter is None else miniter
    maxiter = 200 if maxiter is None else maxiter
    xtol = xtol * size(x0)

    pos = x0
    fun_and_grad, hessp = _prepare_vag_hessp(
        fun, jac, hessp, fun_and_grad=fun_and_grad
    )
    cg_kwargs = {} if cg_kwargs is None else cg_kwargs
    cg_name = name + "CG" if name is not None else None

    nm = "N" if name is None else name

    pp = partial(
        callback,
        _ncg_pretty_print_it,
        hide_strings(name),
        xtol=xtol,
        absdelta=absdelta
    )

    gradnorm = (
        partial(jft_norm, ord=norm_ord)
        if custom_gradnorm is None else custom_gradnorm
    )
    energy, g = fun_and_grad(pos)
    conditional_raise(jnp.isnan(energy), ValueError("energy is Nan"))
    val = {
        "status": -2,
        "iteration": 1,
        "pos": pos,
        "energy": energy,
        "old_energy": old_fval if old_fval is not None else jnp.inf,
        "g": g,
        "nfev": 1,
        "njev": 1,
        "nhev": 0,
    }

    def continue_condition_newton_cg(v):
        return v["status"] < -1

    def single_newton_cg_step(v):
        status, i = v["status"], v["iteration"]
        pos = v["pos"]
        energy, g = v["energy"], v["g"]
        old_energy = v["old_energy"]
        nfev, njev, nhev = v["nfev"], v["njev"], v["nhev"]

        cg_absdelta = jnp.where(
            (~jnp.isinf(old_energy)) & (energy_reduction_factor is not None),
            energy_reduction_factor * (old_energy - energy),
            None if absdelta is None else
            jnp.array(absdelta / 100., dtype=energy.dtype),
        )
        mag_g = jft_norm(g, ord=cg_kwargs.get("norm_ord", 1))
        cg_resnorm = jnp.minimum(
            0.5, jnp.sqrt(mag_g)
        ) * mag_g  # taken from SciPy
        default_kwargs = {
            "absdelta": cg_absdelta,
            "resnorm": cg_resnorm,
            "norm_ord": 1,
            "_raise_nonposdef": False,  # handle non-pos-def
            "name": cg_name
        }
        cg_res = cg(Partial(hessp, pos), g, **{**default_kwargs, **cg_kwargs})
        nat_g, info = cg_res.x, cg_res.info
        nhev += cg_res.nfev
        conditional_raise(
            (info is not None) & (info < 0),
            ValueError("conjugate Gradient failed")
        )

        ret_ls = _line_search_successive_halving(
            pos, energy, g, nat_g, fun_and_grad, hessp, nm
        )
        status = jnp.where(ret_ls["status"] != 0, -1, status)

        # only update values if line search was successful
        old_energy = jnp.where(status < -1, energy, old_energy)
        energy = jnp.where(status < -1, ret_ls["new_energy"], energy)
        energy_diff = jnp.where(status < -1, old_energy - energy, 0.)
        pos = where(status < -1, ret_ls["new_pos"], pos)
        g = where(status < -1, ret_ls["new_g"], g)

        grad_scaling = jnp.where(status < -1, ret_ls["grad_scaling"], 0.)

        nfev += ret_ls["nfev"]
        njev += ret_ls["njev"]
        nhev += ret_ls["nhev"]

        descent_norm = grad_scaling * gradnorm(ret_ls['dd'])
        if name is not None:
            pp(
                i=i,
                grad_scaling=grad_scaling,
                ls_reset=ret_ls["reset"],
                nhev=nhev,
                descent_norm=descent_norm,
                energy=energy,
                energy_diff=energy_diff
            )
        status = jnp.where(jnp.isnan(energy), -1, status)
        conditional_raise(jnp.isnan(energy), ValueError('energy is NaN'))
        min_cond = (ret_ls["iteration"] < 2) & (i > miniter)
        status = jnp.where(
            (absdelta is not None) & (0. <= energy_diff) &
            (energy_diff < absdelta) & min_cond & (status != -1), 0, status
        )
        status = jnp.where(
            (descent_norm <= xtol) & (i > miniter) & (status != -1), 0, status
        )
        status = jnp.where((i == maxiter) & (status < -1), i, status)

        ret = {
            "status": status,
            "iteration": i,
            "pos": pos,
            "energy": energy,
            "old_energy": old_energy,
            "g": g,
            "nfev": nfev,
            "njev": njev,
            "nhev": nhev,
        }
        return ret

    val = while_loop(continue_condition_newton_cg, single_newton_cg_step, val)
    conditional_call(
        val["status"] > 0, logger.error,
        PyTreeString(f"{nm}: Iteration Limit Reached!")
    )
    return OptimizeResults(
        x=val["pos"],
        success=True,
        status=val["status"],
        fun=val["energy"],
        jac=val["g"],
        nit=val["iteration"],
        nfev=val["nfev"],
        njev=val["njev"],
        nhev=val["nhev"]
    )


def _line_search_successive_halving(
    pos, start_energy, g, nat_g, fun_and_grad, hessp, name
):
    from .lax import cond, while_loop

    val = {
        "status": -2,
        "iteration": 0,
        "new_pos": pos,  # placeholder value
        "new_energy": jnp.inf,  # placeholder value
        "new_g": g,  # placeholder value
        "dd": nat_g,  # negative descent direction
        "grad_scaling": 1.,
        "reset": False,
        "nfev": 0,
        "njev": 0,
        "nhev": 0,
    }

    def continue_condition_line_search(val):
        return val["status"] < -1

    def line_search_single_step(val):
        status = val["status"]
        i, reset = val["iteration"], val["reset"]
        grad_scaling, dd = val["grad_scaling"], val["dd"]
        nfev, njev, nhev = val["nfev"], val["njev"], val["nhev"]

        new_pos = pos - grad_scaling * dd
        new_energy, new_g = fun_and_grad(new_pos)
        nfev, njev = nfev + 1, njev + 1

        status = jnp.where(new_energy <= start_energy, 0, status)
        grad_scaling = jnp.where(status < -1, grad_scaling / 2, grad_scaling)

        do_reset = (i == 5) & (status < -1)
        reset = jnp.where(do_reset, True, reset)
        grad_scaling = jnp.where(do_reset, 1., grad_scaling)
        dd = cond(
            do_reset,
            lambda x: vdot(g, g) / g.dot(hessp(pos, g)) * g,
            lambda x: x,
            dd,
        )
        nhev += do_reset

        do_abort = (i == 8) & (status < -1)
        status = jnp.where(do_abort, -1, status)
        conditional_call(
            do_abort,
            logger.error,
            PyTreeString(
                f"{name}: WARNING: Energy would increase; aborting line search."
            ),
        )

        ret = {
            "status": status,
            "iteration": i + 1,
            "new_pos": new_pos,
            "new_energy": new_energy,
            "new_g": new_g,
            "dd": dd,
            "grad_scaling": grad_scaling,
            "reset": reset,
            "nfev": nfev,
            "njev": njev,
            "nhev": nhev,
        }
        return ret

    return while_loop(
        continue_condition_line_search, line_search_single_step, val
    )


class _TrustRegionState(NamedTuple):
    x: Any
    converged: Union[bool, jnp.ndarray]
    status: Union[int, jnp.ndarray]
    fun: Any
    jac: Any
    nfev: Union[int, jnp.ndarray]
    njev: Union[int, jnp.ndarray]
    nhev: Union[int, jnp.ndarray]
    nit: Union[int, jnp.ndarray]
    trust_radius: Union[float, jnp.ndarray]
    jac_magnitude: Union[float, jnp.ndarray]
    old_fval: Union[float, jnp.ndarray]


def _trust_ncg(
    fun=None,
    x0=None,
    *,
    maxiter: Optional[int] = None,
    energy_reduction_factor=0.1,
    old_fval=jnp.nan,
    absdelta=None,
    gtol: float = 1e-4,
    max_trust_radius: Union[float, jnp.ndarray] = 1000.,
    initial_trust_radius: Union[float, jnp.ndarray] = 1.0,
    eta: Union[float, jnp.ndarray] = 0.15,
    subproblem=conjugate_gradient._cg_steihaug_subproblem,
    jac: Optional[Callable] = None,
    hessp: Optional[Callable] = None,
    fun_and_grad: Optional[Callable] = None,
    subproblem_kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None
) -> OptimizeResults:
    from jax.debug import callback

    maxiter = 200 if maxiter is None else maxiter

    status = jnp.where(maxiter == 0, 1, 0)

    if not (0 <= eta < 0.25):
        raise Exception("invalid acceptance stringency")
    # Exception("gradient tolerance must be positive")
    status = jnp.where(gtol < 0., -1, status)
    # Exception("max trust radius must be positive")
    status = jnp.where(max_trust_radius <= 0, -1, status)
    # ValueError("initial trust radius must be positive")
    status = jnp.where(initial_trust_radius <= 0, -1, status)
    # ValueError("initial trust radius must be less than the max trust radius")
    status = jnp.where(initial_trust_radius >= max_trust_radius, -1, status)

    common_dtp = result_type(x0)
    eps = 6. * jnp.finfo(
        common_dtp
    ).eps  # Inspired by SciPy's NewtonCG minimzer

    fun_and_grad, hessp = _prepare_vag_hessp(
        fun, jac, hessp, fun_and_grad=fun_and_grad
    )
    subproblem_kwargs = {} if subproblem_kwargs is None else subproblem_kwargs
    cg_name = name + "SP" if name is not None else None

    f_0, g_0 = fun_and_grad(x0)
    g_0_mag = jft_norm(g_0, ord=subproblem_kwargs.get("norm_ord", 1))
    status = jnp.where(jnp.isfinite(g_0_mag), status, 2)
    init_params = _TrustRegionState(
        converged=False,
        status=status,
        nit=0,
        x=x0,
        fun=f_0,
        jac=g_0,
        jac_magnitude=g_0_mag,
        nfev=1,
        njev=1,
        nhev=0,
        trust_radius=initial_trust_radius,
        old_fval=old_fval
    )

    def pp(arg):
        i = arg["i"]
        msg = (
            "{name}: ↗:{tr:.6e} ⬤:{hit} ∝:{rho:.2e} #∇²:{nhev:02d}"
            "\n{name}: Iteration {i} ⛰:{energy:+.6e} Δ⛰:{energy_diff:.6e}" +
            (" ➽:{absdelta:.6e}" if absdelta is not None else "") +
            ("\n{name}: Iteration Limit Reached" if i == maxiter else "")
        )
        logger.info(msg.format(name=name, **arg))

    def _trust_region_body_f(params: _TrustRegionState) -> _TrustRegionState:
        x_k, g_k, g_k_mag = params.x, params.jac, params.jac_magnitude
        i, f_k, old_fval = params.nit, params.fun, params.old_fval
        tr = params.trust_radius

        i += 1

        if energy_reduction_factor:
            cg_absdelta = energy_reduction_factor * (old_fval - f_k)
        else:
            cg_absdelta = None if absdelta is None else absdelta / 100.
        cg_resnorm = jnp.minimum(0.5, jnp.sqrt(g_k_mag)) * g_k_mag
        # TODO: add an internal success check for future subproblem approaches
        # that might not be solvable
        default_kwargs = {
            "absdelta": cg_absdelta,
            "resnorm": cg_resnorm,
            "trust_radius": tr,
            "norm_ord": 1,
            "name": cg_name
        }
        sub_result = subproblem(
            f_k, g_k, Partial(hessp, x_k), **{
                **default_kwargs,
                **subproblem_kwargs
            }
        )

        pred_f_kp1 = sub_result.pred_f
        x_kp1 = x_k + sub_result.step
        f_kp1, g_kp1 = fun_and_grad(x_kp1)

        actual_reduction = f_k - f_kp1
        pred_reduction = f_k - pred_f_kp1

        # update the trust radius according to the actual/predicted ratio
        rho = actual_reduction / pred_reduction
        tr_kp1 = jnp.where(rho < 0.25, tr * 0.25, tr)
        tr_kp1 = jnp.where(
            (rho > 0.75) & sub_result.hits_boundary,
            jnp.minimum(2. * tr, max_trust_radius), tr_kp1
        )

        # compute norm to check for convergence
        g_kp1_mag = jft_norm(g_kp1, ord=subproblem_kwargs.get("norm_ord", 1))

        # if the ratio is high enough then accept the proposed step
        f_kp1, x_kp1, g_kp1, g_kp1_mag = where(
            rho > eta, (f_kp1, x_kp1, g_kp1, g_kp1_mag),
            (f_k, x_k, g_k, g_k_mag)
        )

        # Check whether we arrived at the float precision
        energy_eps = eps * jnp.abs(f_kp1)
        converged = (actual_reduction
                     <= energy_eps) & (actual_reduction > -energy_eps)

        converged |= g_kp1_mag < gtol
        if absdelta:
            converged |= (rho > eta) & (actual_reduction
                                        > 0.) & (actual_reduction < absdelta)

        status = jnp.where(converged, 0, params.status)
        status = jnp.where(i >= maxiter, 1, status)
        status = jnp.where(pred_reduction <= 0, 2, status)
        params = _TrustRegionState(
            converged=converged,
            nit=i,
            x=x_kp1,
            fun=f_kp1,
            jac=g_kp1,
            jac_magnitude=g_kp1_mag,
            nfev=params.nfev + sub_result.nfev + 1,
            njev=params.njev + sub_result.njev + 1,
            nhev=params.nhev + sub_result.nhev,
            trust_radius=tr_kp1,
            status=status,
            old_fval=f_k
        )
        if name is not None:
            printable_state = {
                "i": i,
                "energy": params.fun,
                "energy_diff": actual_reduction,
                "maxiter": maxiter,
                "absdelta": absdelta,
                "tr": params.trust_radius,
                "rho": rho,
                "nhev": params.nhev,
                "hit": sub_result.hits_boundary
            }
            callback(pp, printable_state)
        return params

    def _trust_region_cond_f(params: _TrustRegionState) -> bool:
        return jnp.logical_not(params.converged) & (params.status == 0)

    state = lax.while_loop(
        _trust_region_cond_f, _trust_region_body_f, init_params
    )

    return OptimizeResults(
        success=state.converged & (state.status == 0),
        nit=state.nit,
        x=state.x,
        fun=state.fun,
        jac=state.jac,
        nfev=state.nfev,
        njev=state.njev,
        nhev=state.nhev,
        jac_magnitude=state.jac_magnitude,
        trust_radius=state.trust_radius,
        status=state.status
    )


def trust_ncg(fun=None, x0=None, *args, **kwargs):
    if x0 is not None:
        assert_arithmetics(x0)
    return _trust_ncg(fun, x0, *args, **kwargs).x


def minimize(
    fun: Optional[Callable[..., float]],
    x0,
    args: Tuple = (),
    *,
    method: str,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None
) -> OptimizeResults:
    """Minimize fun."""
    assert_arithmetics(x0)
    if options is None:
        options = {}
    if not isinstance(args, tuple):
        te = f"args argument must be a tuple, got {type(args)!r}"
        raise TypeError(te)

    fun_with_args = fun
    if args:
        fun_with_args = lambda x: fun(x, *args)

    if tol is not None:
        raise ValueError("use solver-specific options")

    if method.lower() in ('newton-cg', 'newtoncg', 'ncg'):
        return _newton_cg(fun_with_args, x0, **options)
    elif method.lower() in ('trust-ncg', 'trustncg'):
        return _trust_ncg(fun_with_args, x0, **options)

    raise ValueError(f"method {method} not recognized")
