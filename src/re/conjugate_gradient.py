# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from datetime import datetime
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
from jax import numpy as jnp

from .logger import logger
from .misc import doc_from
from .tree_math import assert_arithmetics
from .tree_math import norm as jft_norm
from .tree_math import result_type, size, vdot, where, zeros_like

HessVP = Callable[[jnp.ndarray], jnp.ndarray]

N_RESET = 20


class CGResults(NamedTuple):
    x: jnp.ndarray
    nit: Union[int, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]  # number of matrix-evaluations
    info: Union[int, jnp.ndarray]
    success: Union[bool, jnp.ndarray]


def cg(mat, j, x0=None, *args, **kwargs) -> Tuple[Any, Union[int, jnp.ndarray]]:
    """Solve `mat(x) = j` using Conjugate Gradient. `mat` must be callable and
    represent a hermitian, positive definite matrix.

    Notes
    -----
    If set, the parameters `absdelta` and `resnorm` always take precedence over
    `tol` and `atol`.
    """
    assert_arithmetics(j)
    if x0 is not None:
        assert_arithmetics(x0)
    cg_res = _cg(mat, j, x0, *args, **kwargs)
    return cg_res.x, cg_res.info


@doc_from(cg)
def static_cg(mat, j, x0=None, *args, **kwargs):
    assert_arithmetics(j)
    if x0 is not None:
        assert_arithmetics(x0)
    cg_res = _static_cg(mat, j, x0, *args, **kwargs)
    return cg_res.x, cg_res.info


def _cg_pretty_print_it(
    name,
    i,
    *,
    energy,
    energy_diff,
    absdelta=None,
    norm=None,
    resnorm=None,
    maxiter=None
):
    if maxiter is not None and i == maxiter:
        i_str = "✖" * len(str(i)) + f" ({i})"
    else:
        i_str = str(i)
    msg = f"{name}: Iteration {i_str} ⛰:{energy:+.4e} Δ⛰:{energy_diff:.4e}"
    msg += f" ➽:{absdelta:.4e}" if absdelta is not None else ""
    if norm is not None and resnorm is not None:
        msg += f" |∇|:{norm:.4e} ➽:{resnorm:.4e}"
    logger.info(msg)


# Taken from nifty
def _cg(
    mat,
    j,
    x0=None,
    *,
    absdelta=None,
    resnorm=None,
    norm_ord=None,
    tol=1e-5,  # taken from SciPy's linalg.cg
    atol=0.,
    miniter=None,
    maxiter=None,
    name=None,
    time_threshold=None,
    _raise_nonposdef=True
) -> CGResults:
    norm_ord = 2 if norm_ord is None else norm_ord  # TODO: change to 1
    maxiter_fallback = 20 * size(j)  # taken from SciPy's NewtonCG minimzer
    miniter = min(
        (6, maxiter if maxiter is not None else maxiter_fallback)
    ) if miniter is None else miniter
    maxiter = max(
        (min((200, maxiter_fallback)), miniter)
    ) if maxiter is None else maxiter

    if absdelta is None and resnorm is None:  # fallback convergence criterion
        resnorm = jnp.maximum(tol * jft_norm(j, ord=norm_ord), atol)

    common_dtp = result_type(j)
    eps = 6. * jnp.finfo(common_dtp).eps  # taken from SciPy's NewtonCG minimzer
    tiny = 6. * jnp.finfo(common_dtp).tiny

    if x0 is None:
        pos = zeros_like(j)
        r = -j
        d = r
        # energy = .5xT M x - xT j
        energy = 0.
        nfev = 0
    else:
        pos = x0
        r = mat(pos) - j
        d = r
        energy = float(vdot((r - j) / 2, pos))
        nfev = 1
    previous_gamma = float(vdot(r, r))

    info = -1
    i = 0
    energy_diff = jnp.inf
    norm = None
    pp = partial(
        _cg_pretty_print_it,
        name,
        absdelta=absdelta,
        resnorm=resnorm,
        maxiter=maxiter
    )
    if name is not None:
        if resnorm is not None:
            norm = jft_norm(r, ord=norm_ord)
        else:
            norm = None
        pp(i, energy=energy, energy_diff=energy_diff, norm=norm)

    if previous_gamma == 0:
        info = 0
        return CGResults(x=pos, info=info, nit=0, nfev=nfev, success=True)

    for i in range(1, maxiter + 1):
        q = mat(d)
        nfev += 1

        curv = float(vdot(d, q))
        if curv == 0.:
            if _raise_nonposdef:
                nm = "CG" if name is None else name
                raise ValueError(f"{nm}: zero curvature")
            info = 0
            break
        elif curv < 0.:
            if _raise_nonposdef:
                nm = "CG" if name is None else name
                raise ValueError(f"{nm}: negative curvature")
            if i > 1:
                info = 0
                break
            else:
                pos = previous_gamma / (-curv) * (-j)
                info = 0
                break
        alpha = previous_gamma / curv
        pos = pos - alpha * d
        if i % N_RESET == 0:
            r = mat(pos) - j
            nfev += 1
        else:
            r = r - q * alpha
        gamma = float(vdot(r, r))
        if time_threshold is not None and datetime.now() > time_threshold:
            info = i
            break
        if gamma >= 0. and gamma <= tiny:
            nm = "CG" if name is None else name
            logger.warning(f"{nm}: gamma=0, converged!")
            info = 0
            break
        if resnorm is not None:
            norm = float(jft_norm(r, ord=norm_ord))
            if norm < resnorm and i >= miniter:
                info = 0
                break
        else:
            norm = None
        new_energy = float(vdot((r - j) / 2, pos))
        energy_diff = energy - new_energy
        neg_energy_eps = -eps * jnp.abs(new_energy)
        if energy_diff < neg_energy_eps:
            nm = "CG" if name is None else name
            if _raise_nonposdef:
                raise ValueError(f"{nm}: WARNING: energy increased")
            logger.error(f"{nm}: WARNING: energy increased")
            info = i
            break
        if absdelta is not None and energy_diff < absdelta and i >= miniter:
            info = 0
            break
        energy = new_energy
        d = d * max(0, gamma / previous_gamma) + r
        previous_gamma = gamma

        if name is not None:
            pp(i, energy=energy, energy_diff=energy_diff, norm=norm)

    if name is not None and info != -1:
        # only print if loop was terminated via `break` otherwise everything is
        pp(i, energy=energy, energy_diff=energy_diff, norm=norm)

    info = i if info == -1 else info
    return CGResults(x=pos, info=info, nit=i, nfev=nfev, success=info == 0)


def _static_cg(
    mat,
    j,
    x0=None,
    *,
    absdelta=None,
    resnorm=None,
    norm_ord=None,
    tol=1e-5,  # taken from SciPy's linalg.cg
    atol=0.,
    miniter=None,
    maxiter=None,
    name=None,
    _raise_nonposdef=True,
    **kwargs
) -> CGResults:
    from jax.debug import callback
    from jax.lax import cond, while_loop

    # While in general it is strongly discouraged to use `jit` inside a
    # function, we do it here to avoid the overhead of jitting through `mat`
    # multiple times within the CG loop. This is safe because the call to
    # `while_loop` in CG implies a JIT anyways.
    mat = jax.jit(mat)

    norm_ord = 2 if norm_ord is None else norm_ord  # TODO: change to 1
    maxiter_fallback = 20 * size(j)  # taken from SciPy's NewtonCG minimzer
    miniter = jnp.minimum(
        6, maxiter if maxiter is not None else maxiter_fallback
    ) if miniter is None else miniter
    maxiter = jnp.maximum(
        jnp.minimum(200, maxiter_fallback), miniter
    ) if maxiter is None else maxiter

    if absdelta is None and resnorm is None:  # fallback convergence criterion
        resnorm = jnp.maximum(tol * jft_norm(j, ord=norm_ord), atol)

    common_dtp = result_type(j)
    eps = 6. * jnp.finfo(common_dtp).eps  # taken from SciPy's NewtonCG minimzer
    tiny = 6. * jnp.finfo(common_dtp).tiny

    def pp(arg):
        _cg_pretty_print_it(name, **arg)

    def continue_condition(v):
        return v["info"] < -1

    def cg_single_step(v):
        info = v["info"]
        pos, r, d, i = v["pos"], v["r"], v["d"], v["iteration"]
        previous_gamma, previous_energy = v["gamma"], v["energy"]

        i += 1

        q = mat(d)
        curv = vdot(d, q)
        alpha = previous_gamma / curv
        # ValueError("implausible or zero curvature in conjugate gradient")
        info = jnp.where(curv <= 0., jnp.where(_raise_nonposdef, -1, 0), info)
        alpha = jnp.where((curv <= 0.) & (not _raise_nonposdef), 0., alpha)
        pos = pos - alpha * d
        pos = where(
            (curv < 0.) & (not _raise_nonposdef) & (i <= 1),
            previous_energy / (-curv) * (-j),
            pos,
        )
        r = cond(
            (i % N_RESET == 0) & (info < -1),
            lambda x: mat(x["pos"]) - x["j"],
            lambda x: x["r"] - x["q"] * x["alpha"],
            {
                "pos": pos,
                "j": j,
                "r": r,
                "q": q,
                "alpha": alpha
            },
        )
        gamma = vdot(r, r)

        info = jnp.where(
            (gamma >= 0.) & (gamma <= tiny) & (info != -1), 0, info
        )
        if resnorm is not None:
            norm = jft_norm(r, ord=norm_ord)
            info = jnp.where(
                (norm < resnorm) & (i >= miniter) & (info != -1), 0, info
            )
        else:
            norm = None
        energy = vdot((r - j) / 2, pos)
        energy_diff = previous_energy - energy
        neg_energy_eps = -eps * jnp.abs(energy)
        # print(f"energy increased", file=sys.stderr)
        info = jnp.where(
            energy_diff < neg_energy_eps,
            jnp.where(_raise_nonposdef, -1, i),
            info,
        )
        if absdelta is not None:
            info = jnp.where(
                (energy_diff < absdelta) & (i >= miniter) & (info != -1), 0,
                info
            )
        info = jnp.where((i >= maxiter) & (info != -1), i, info)

        d = d * jnp.maximum(0, gamma / previous_gamma) + r

        if name is not None:
            printable_state = {
                "i": i,
                "energy": energy,
                "energy_diff": energy_diff,
                "absdelta": absdelta,
                "norm": norm,
                "resnorm": resnorm,
                "maxiter": maxiter
            }
            callback(pp, printable_state)

        ret = {
            "info": info,
            "pos": pos,
            "r": r,
            "d": d,
            "iteration": i,
            "gamma": gamma,
            "energy": energy
        }
        return ret

    if x0 is None:
        pos = zeros_like(j)
        r = -j
        d = r
        nfev = 0
    else:
        pos = x0
        r = mat(pos) - j
        d = r
        nfev = 1
    # energy = .5xT M x - xT j
    energy = jnp.array(0.) if x0 is None else vdot((r - j) / 2, pos)

    gamma = vdot(r, r)
    val = {
        "info": jnp.array(-2, dtype=int),
        "pos": pos,
        "r": r,
        "d": d,
        "iteration": jnp.array(0),
        "gamma": gamma,
        "energy": energy
    }
    # Finish early if already converged in the initial iteration
    val["info"] = jnp.where(gamma == 0., 0, val["info"])

    if name is not None:
        if resnorm is not None:
            norm = jft_norm(r, ord=norm_ord)
        else:
            norm = None
        printable_state = {
            "i": 0,
            "energy": energy,
            "energy_diff": jnp.inf,
            "absdelta": absdelta,
            "norm": norm,
            "resnorm": resnorm,
            "maxiter": maxiter
        }
        callback(pp, printable_state)

    val = while_loop(continue_condition, cg_single_step, val)

    i = val["iteration"]
    info = val["info"]
    nfev += i + i // N_RESET
    return CGResults(
        x=val["pos"], info=info, nit=i, nfev=nfev, success=info == 0
    )


# The following is code adapted from Nicholas Mancuso to work with pytrees
class _QuadSubproblemResult(NamedTuple):
    step: jnp.ndarray
    hits_boundary: Union[bool, jnp.ndarray]
    pred_f: Union[float, jnp.ndarray]
    nit: Union[int, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    njev: Union[int, jnp.ndarray]
    nhev: Union[int, jnp.ndarray]
    success: Union[bool, jnp.ndarray]


class _CGSteihaugState(NamedTuple):
    z: jnp.ndarray
    r: jnp.ndarray
    d: jnp.ndarray
    step: jnp.ndarray
    energy: Union[None, float, jnp.ndarray]
    hits_boundary: Union[bool, jnp.ndarray]
    done: Union[bool, jnp.ndarray]
    nit: Union[int, jnp.ndarray]
    nhev: Union[int, jnp.ndarray]


def second_order_approx(
    p: jnp.ndarray,
    cur_val: Union[float, jnp.ndarray],
    g: jnp.ndarray,
    hessp_at_xk: HessVP,
) -> Union[float, jnp.ndarray]:
    return cur_val + vdot(g, p) + 0.5 * vdot(p, hessp_at_xk(p))


def get_boundaries_intersections(
    z: jnp.ndarray, d: jnp.ndarray, trust_radius: Union[float, jnp.ndarray]
):  # Adapted from SciPy
    """Solve the scalar quadratic equation ||z + t d|| == trust_radius.

    This is like a line-sphere intersection.

    Return the two values of t, sorted from low to high.
    """
    a = vdot(d, d)
    b = 2 * vdot(z, d)
    c = vdot(z, z) - trust_radius**2
    sqrt_discriminant = jnp.sqrt(b * b - 4 * a * c)

    # The following calculation is mathematically
    # equivalent to:
    # ta = (-b - sqrt_discriminant) / (2*a)
    # tb = (-b + sqrt_discriminant) / (2*a)
    # but produce smaller round off errors.
    # Look at Matrix Computation p.97
    # for a better justification.
    aux = b + jnp.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux

    ra, rb = where(ta < tb, (ta, tb), (tb, ta))
    return (ra, rb)


def _cg_steihaug_subproblem(
    cur_val: Union[float, jnp.ndarray],
    g: jnp.ndarray,
    hessp_at_xk: HessVP,
    *,
    trust_radius: Union[float, jnp.ndarray],
    tr_norm_ord: Union[None, int, float, jnp.ndarray] = None,
    resnorm: Optional[float],
    absdelta: Optional[float] = None,
    norm_ord: Union[None, int, float, jnp.ndarray] = None,
    miniter: Union[None, int] = None,
    maxiter: Union[None, int] = None,
    name=None
) -> _QuadSubproblemResult:
    """
    Solve the subproblem using a conjugate gradient method.

    Parameters
    ----------
    cur_val : Union[float, jnp.ndarray]
      Objective value evaluated at the current state.
    g : jnp.ndarray
      Gradient value evaluated at the current state.
    hessp_at_xk: Callable
      Function that accepts a proposal vector and computes the result of a
      Hessian-vector product.
    trust_radius : float
      Upper bound on how large a step proposal can be.
    tr_norm_ord : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
      Order of the norm for computing the length of the next step.
    norm_ord : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
      Order of the norm for testing convergence.

    Returns
    -------
    result : _QuadSubproblemResult
      Contains the step proposal, whether it is at radius boundary, and
      meta-data regarding function calls and successful convergence.

    Notes
    -----
    This is algorithm (7.2) of Nocedal and Wright 2nd edition.
    Only the function that computes the Hessian-vector product is required.
    The Hessian itself is not required, and the Hessian does
    not need to be positive semidefinite.
    """
    from jax.debug import callback
    from jax.lax import switch, while_loop

    tr_norm_ord = jnp.inf if tr_norm_ord is None else tr_norm_ord  # taken from JAX
    norm_ord = 2 if norm_ord is None else norm_ord  # TODO: change to 1
    maxiter_fallback = 20 * size(g)  # taken from SciPy's NewtonCG minimzer
    miniter = jnp.minimum(
        6, maxiter if maxiter is not None else maxiter_fallback
    ) if miniter is None else miniter
    maxiter = jnp.maximum(
        jnp.minimum(200, maxiter_fallback), miniter
    ) if maxiter is None else maxiter

    common_dtp = result_type(g)
    eps = 6. * jnp.finfo(
        common_dtp
    ).eps  # Inspired by SciPy's NewtonCG minimzer

    # second-order Taylor series approximation at the current values, gradient,
    # and hessian
    soa = partial(
        second_order_approx, cur_val=cur_val, g=g, hessp_at_xk=hessp_at_xk
    )

    def pp(arg):
        msg = (
            "{name}: |∇|:{r_norm:.6e} ➽:{resnorm:.6e} ↗:{tr:.6e}"
            " ☞:{case:1d} #∇²:{nhev:02d}"
            "\n{name}: Iteration {i} ⛰:{energy:+.6e} Δ⛰:{energy_diff:.6e}" +
            (" ➽:{absdelta:.6e}" if arg["absdelta"] is not None else "") + (
                "\n{name}: Iteration Limit Reached"
                if arg["i"] == arg["maxiter"] else ""
            )
        )
        logger.info(msg.format(name=name, **arg))

    # helpers for internal switches in the main CGSteihaug logic
    def noop(
        param: Tuple[_CGSteihaugState, Union[float, jnp.ndarray]]
    ) -> _CGSteihaugState:
        iterp, z_next = param
        return iterp

    def step1(
        param: Tuple[_CGSteihaugState, Union[float, jnp.ndarray]]
    ) -> _CGSteihaugState:
        iterp, z_next = param
        z, d, nhev = iterp.z, iterp.d, iterp.nhev

        ta, tb = get_boundaries_intersections(z, d, trust_radius)
        pa = z + ta * d
        pb = z + tb * d
        p_boundary = where(soa(pa) < soa(pb), pa, pb)
        return iterp._replace(
            step=p_boundary, nhev=nhev + 2, hits_boundary=True, done=True
        )

    def step2(
        param: Tuple[_CGSteihaugState, Union[float, jnp.ndarray]]
    ) -> _CGSteihaugState:
        iterp, z_next = param
        z, d = iterp.z, iterp.d

        ta, tb = get_boundaries_intersections(z, d, trust_radius)
        p_boundary = z + tb * d
        return iterp._replace(step=p_boundary, hits_boundary=True, done=True)

    def step3(
        param: Tuple[_CGSteihaugState, Union[float, jnp.ndarray]]
    ) -> _CGSteihaugState:
        iterp, z_next = param
        return iterp._replace(step=z_next, hits_boundary=False, done=True)

    # initialize the step
    p_origin = zeros_like(g)

    # init the state for the first iteration
    z = p_origin
    r = g
    d = -r
    energy = 0.
    init_param = _CGSteihaugState(
        z=z,
        r=r,
        d=d,
        step=p_origin,
        energy=energy,
        hits_boundary=False,
        done=maxiter == 0,
        nit=0,
        nhev=0
    )

    # Search for the min of the approximation of the objective function.
    def body_f(iterp: _CGSteihaugState) -> _CGSteihaugState:
        z, r, d = iterp.z, iterp.r, iterp.d
        energy, nit = iterp.energy, iterp.nit

        nit += 1

        Bd = hessp_at_xk(d)
        dBd = vdot(d, Bd)

        r_squared = vdot(r, r)
        alpha = r_squared / dBd
        z_next = z + alpha * d

        r_next = r + alpha * Bd
        r_next_squared = vdot(r_next, r_next)

        beta_next = r_next_squared / r_squared
        d_next = -r_next + beta_next * d

        accept_z_next = nit >= maxiter
        if norm_ord == 2:
            r_next_norm = jnp.sqrt(r_next_squared)
        else:
            r_next_norm = jft_norm(r_next, ord=norm_ord)
        accept_z_next |= r_next_norm < resnorm
        # Relative to a plain CG, `z_next` is negative
        energy_next = vdot((r_next + g) / 2, z_next)
        energy_diff = energy - energy_next
        if absdelta is not None:
            neg_energy_eps = -eps * jnp.abs(energy)
            accept_z_next |= (energy_diff >= neg_energy_eps
                             ) & (energy_diff < absdelta) & (nit >= miniter)

        # include a junk switch to catch the case where none should be executed
        z_next_norm = jft_norm(z_next, ord=tr_norm_ord)
        index = jnp.argmax(
            jnp.array(
                [False, dBd <= 0, z_next_norm >= trust_radius, accept_z_next]
            )
        )
        iterp = switch(index, [noop, step1, step2, step3], (iterp, z_next))

        iterp = iterp._replace(
            z=z_next,
            r=r_next,
            d=d_next,
            energy=energy_next,
            nhev=iterp.nhev + 1,
            nit=nit
        )
        if name is not None:
            printable_state = {
                "i": nit,
                "energy": iterp.energy,
                "energy_diff": energy_diff,
                "absdelta": absdelta,
                "tr": trust_radius,
                "r_norm": r_next_norm,
                "resnorm": resnorm,
                "nhev": iterp.nhev,
                "case": index,
                "maxiter": maxiter
            }
            callback(pp, printable_state)

        return iterp

    def cond_f(iterp: _CGSteihaugState) -> bool:
        return jnp.logical_not(iterp.done)

    # perform inner optimization to solve the constrained
    # quadratic subproblem using cg
    result = while_loop(cond_f, body_f, init_param)

    pred_f = soa(result.step)
    result = _QuadSubproblemResult(
        step=result.step,
        hits_boundary=result.hits_boundary,
        pred_f=pred_f,
        nit=result.nit,
        nfev=0,
        njev=0,
        nhev=result.nhev + 1,
        success=True
    )

    return result
