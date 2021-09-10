import sys
from datetime import datetime
from functools import partial
from jax import numpy as np
from jax import lax

from typing import Callable, NamedTuple, Optional, Tuple, Union

from .forest_util import common_type, size, where, zeros_like
from .forest_util import norm as jft_norm
from .sugar import doc_from, sum_of_squares

HessVP = Callable[[np.ndarray], np.ndarray]

N_RESET = 20


# Taken from nifty
def cg(
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
    time_threshold=None
):
    """Solve `mat(x) = j` using Conjugate Gradient. `mat` must be callable and
    represent a hermitian, positive definite matrix.

    Notes
    -----
    If set, the parameters `absdelta` and `resnorm` always take precedence over
    `tol` and `atol`.
    """
    norm_ord = 2 if norm_ord is None else norm_ord  # TODO: change to 1
    maxiter_fallback = 20 * size(j)  # taken from SciPy's NewtonCG minimzer
    miniter = min(
        (6, maxiter if maxiter is not None else maxiter_fallback)
    ) if miniter is None else miniter
    maxiter = max(
        (min((200, maxiter_fallback)), miniter)
    ) if maxiter is None else maxiter

    if absdelta is None and resnorm is None:  # fallback convergence criterion
        resnorm = np.maximum(tol * jft_norm(j, ord=norm_ord, ravel=True), atol)

    common_dtp = common_type(j)
    eps = 6. * np.finfo(common_dtp).eps  # taken from SciPy's NewtonCG minimzer

    if x0 is None:
        pos = zeros_like(j)
        r = -j
        d = r
        # energy = .5xT M x - xT j
        energy = 0.
    else:
        pos = x0
        r = mat(pos) - j
        d = r
        energy = float(((r - j) / 2).dot(pos))
    previous_gamma = float(sum_of_squares(r))
    if previous_gamma == 0:
        info = 0
        return pos, info

    info = -1
    i = 0
    for i in range(1, maxiter + 1):
        if name is not None:
            print(f"{name}: Iteration {i} ⛰:{energy:+.6e}", file=sys.stderr)
        q = mat(d)
        curv = float(d.dot(q))
        if curv == 0.:
            raise ValueError("zero curvature in conjugate gradient")
        alpha = previous_gamma / curv
        if alpha < 0:
            raise ValueError("implausible gradient scaling `alpha < 0`")
        pos = pos - alpha * d
        if i % N_RESET == 0:
            r = mat(pos) - j
        else:
            r = r - q * alpha
        gamma = float(sum_of_squares(r))
        if time_threshold is not None and datetime.now() > time_threshold:
            info = i
            return pos, info
        if gamma == 0:
            nm = "CG" if name is None else name
            print(f"{nm}: gamma=0, converged!", file=sys.stderr)
            info = 0
            return pos, info
        if resnorm is not None:
            norm = float(jft_norm(r, ord=norm_ord, ravel=True))
            if name is not None:
                msg = f"{name}: |∇|:{norm:.6e} 🞋:{resnorm:.6e}"
                print(msg, file=sys.stderr)
            if norm < resnorm and i >= miniter:
                info = 0
                return pos, info
        new_energy = float(((r - j) / 2).dot(pos))
        if absdelta is not None:
            energy_diff = energy - new_energy
            if name is not None:
                msg = f"{name}: Δ⛰:{energy_diff:.6e} 🞋:{absdelta:.6e}"
                print(msg, file=sys.stderr)
            basically_zero = -eps * np.abs(new_energy)
            if energy_diff < basically_zero:
                nm = "CG" if name is None else name
                print(f"{nm}: WARNING: Energy increased", file=sys.stderr)
                return pos, -1
            if basically_zero <= energy_diff < absdelta and i >= miniter:
                info = 0
                return pos, info
        energy = new_energy
        d = d * max(0, gamma / previous_gamma) + r
        previous_gamma = gamma
    else:
        nm = "CG" if name is None else name
        print(f"{nm}: Iteration Limit Reached", file=sys.stderr)
        info = i
    return pos, info


@doc_from(cg)
def static_cg(
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
    **kwargs
):
    from jax.lax import cond, while_loop

    norm_ord = 2 if norm_ord is None else norm_ord  # TODO: change to 1
    maxiter_fallback = 20 * size(j)  # taken from SciPy's NewtonCG minimzer
    miniter = min(
        (6, maxiter if maxiter is not None else maxiter_fallback)
    ) if miniter is None else miniter
    maxiter = max(
        (min((200, maxiter_fallback)), miniter)
    ) if maxiter is None else maxiter

    if absdelta is None and resnorm is None:  # fallback convergence criterion
        resnorm = np.maximum(tol * jft_norm(j, ord=norm_ord, ravel=True), atol)

    common_dtp = common_type(j)
    eps = 6. * np.finfo(common_dtp).eps  # Inspired by SciPy's NewtonCG minimzer

    def continue_condition(v):
        return v["info"] < -1

    def cg_single_step(v):
        info = v["info"]
        pos, r, d, i = v["pos"], v["r"], v["d"], v["iteration"]
        previous_gamma, previous_energy = v["gamma"], v["energy"]

        i += 1

        if name is not None:
            msg = f"{name}: Iteration {v['iteration']!r} ⛰:{previous_energy!r}"
            print(msg, file=sys.stderr)

        q = mat(d)
        curv = d.dot(q)
        # ValueError("zero curvature in conjugate gradient")
        info = np.where(curv == 0., -1, info)
        alpha = previous_gamma / curv
        # ValueError("implausible gradient scaling `alpha < 0`")
        info = np.where(alpha < 0., -1, info)
        pos = pos - alpha * d
        r = cond(
            i % N_RESET == 0, lambda x: mat(x["pos"]) - x["j"],
            lambda x: x["r"] - x["q"] * x["alpha"], {
                "pos": pos,
                "j": j,
                "r": r,
                "q": q,
                "alpha": alpha
            }
        )
        gamma = sum_of_squares(r)

        info = np.where((gamma == 0.) & (info != -1), 0, info)
        if resnorm is not None:
            norm = jft_norm(r, ord=norm_ord, ravel=True)
            if name is not None:
                msg = f"{name}: |∇|:{norm!r} 🞋:{resnorm!r}"
                print(msg, file=sys.stderr)
            info = np.where(
                (norm < resnorm) & (i >= miniter) & (info != -1), 0, info
            )
        # Do not compute the energy if we do not check `absdelta`
        if absdelta is not None or name is not None:
            energy = ((r - j) / 2).dot(pos)
        else:
            energy = previous_energy
        if absdelta is not None:
            energy_diff = previous_energy - energy
            if name is not None:
                msg = f"{name}: Δ⛰:{energy_diff!r} 🞋:{absdelta!r}"
                print(msg, file=sys.stderr)
            basically_zero = -eps * np.abs(energy)
            # print(f"{nm}: WARNING: Energy increased", file=sys.stderr)
            info = np.where(energy_diff < basically_zero, -1, info)
            info = np.where(
                (energy_diff >= basically_zero) & (energy_diff < absdelta) &
                (i >= miniter) & (info != -1), 0, info
            )
        info = np.where((i >= maxiter) & (info != -1), i, info)

        d = d * np.maximum(0, gamma / previous_gamma) + r

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
    else:
        pos = x0
        r = mat(pos) - j
        d = r
    energy = None
    if absdelta is not None or name is not None:
        if x0 is None:
            # energy = .5xT M x - xT j
            energy = np.array(0.)
        else:
            energy = ((r - j) / 2).dot(pos)

    gamma = sum_of_squares(r)
    val = {
        "info": np.array(-2, dtype=int),
        "pos": pos,
        "r": r,
        "d": d,
        "iteration": np.array(0),
        "gamma": gamma,
        "energy": energy
    }
    # Finish early if already converged in the initial iteration
    val["info"] = np.where(gamma == 0., 0, val["info"])

    val = while_loop(continue_condition, cg_single_step, val)

    return val["pos"], val["info"]


# The following is code adapted from Nicholas Mancuso to work with pytrees
class _QuadSubproblemResult(NamedTuple):
    step: np.ndarray
    hits_boundary: Union[bool, np.ndarray]
    pred_f: Union[float, np.ndarray]
    nfev: Union[int, np.ndarray]
    njev: Union[int, np.ndarray]
    nhev: Union[int, np.ndarray]
    success: Union[bool, np.ndarray]


class _CGSteihaugState(NamedTuple):
    z: np.ndarray
    r: np.ndarray
    d: np.ndarray
    step: np.ndarray
    energy: Union[None, float, np.ndarray]
    hits_boundary: Union[bool, np.ndarray]
    done: Union[bool, np.ndarray]
    nit: Union[int, np.ndarray]
    nhev: Union[int, np.ndarray]


def second_order_approx(
    p: np.ndarray,
    cur_val: Union[float, np.ndarray],
    g: np.ndarray,
    hessp_at_xk: HessVP,
) -> Union[float, np.ndarray]:
    return cur_val + g.dot(p) + 0.5 * p.dot(hessp_at_xk(p))


def get_boundaries_intersections(
    z: np.ndarray, d: np.ndarray, trust_radius: Union[float, np.ndarray]
):  # Adapted from SciPy
    """Solve the scalar quadratic equation ||z + t d|| == trust_radius.

    This is like a line-sphere intersection.

    Return the two values of t, sorted from low to high.
    """
    a = d.dot(d)
    b = 2 * z.dot(d)
    c = z.dot(z) - trust_radius**2
    sqrt_discriminant = np.sqrt(b * b - 4 * a * c)

    # The following calculation is mathematically
    # equivalent to:
    # ta = (-b - sqrt_discriminant) / (2*a)
    # tb = (-b + sqrt_discriminant) / (2*a)
    # but produce smaller round off errors.
    # Look at Matrix Computation p.97
    # for a better justification.
    aux = b + np.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux

    ra, rb = where(ta < tb, (ta, tb), (tb, ta))
    return (ra, rb)


def _cg_steihaug_subproblem(
    cur_val: Union[float, np.ndarray],
    g: np.ndarray,
    hessp_at_xk: HessVP,
    *,
    trust_radius: Union[float, np.ndarray],
    resnorm: Optional[float],
    absdelta: Optional[float] = None,
    norm_ord: Union[None, int, float, np.ndarray] = None,
    miniter: Union[None, int] = None,
    maxiter: Union[None, int] = None,
    _mag_g: Union[None, float, np.ndarray] = None,
) -> _QuadSubproblemResult:
    """
    Solve the subproblem using a conjugate gradient method.

    Parameters
    ----------
    cur_val : Union[float, np.ndarray]
      Objective value evaluated at the current state.
    g : np.ndarray
      Gradient value evaluated at the current state.
    hessp_at_xk: Callable
      Function that accepts a proposal vector and computes the result of a
      Hessian-vector product.
    trust_radius : float
      Upper bound on how large a step proposal can be.
    norm_ord : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
      Order of the norm. inf means jax.numpy’s inf object. The default is 2.
    _mag_g : Union[float, np.ndarray]
      The magnitude of the gradient `g` using norm_ord=`norm_ord`.

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
    norm_ord = 2 if norm_ord is None else norm_ord  # taken from SciPy
    maxiter_fallback = 20 * size(g)  # taken from SciPy's NewtonCG minimzer
    miniter = min(
        (6, maxiter if maxiter is not None else maxiter_fallback)
    ) if miniter is None else miniter
    maxiter = max(
        (min((200, maxiter_fallback)), miniter)
    ) if maxiter is None else maxiter

    if resnorm is None:
        mag_g = jft_norm(g, ord=norm_ord) if _mag_g is None else _mag_g
        resnorm = np.minimum(0.5, np.sqrt(mag_g)) * mag_g

    common_dtp = common_type(g)
    eps = 6. * np.finfo(common_dtp).eps  # Inspired by SciPy's NewtonCG minimzer

    # second-order Taylor series approximation at the current values, gradient,
    # and hessian
    soa = partial(
        second_order_approx, cur_val=cur_val, g=g, hessp_at_xk=hessp_at_xk
    )

    # helpers for internal switches in the main CGSteihaug logic
    def noop(
        param: Tuple[_CGSteihaugState, Union[float, np.ndarray]]
    ) -> _CGSteihaugState:
        iterp, z_next = param
        return iterp

    def step1(
        param: Tuple[_CGSteihaugState, Union[float, np.ndarray]]
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
        param: Tuple[_CGSteihaugState, Union[float, np.ndarray]]
    ) -> _CGSteihaugState:
        iterp, z_next = param
        z, d = iterp.z, iterp.d

        ta, tb = get_boundaries_intersections(z, d, trust_radius)
        p_boundary = z + tb * d
        return iterp._replace(step=p_boundary, hits_boundary=True, done=True)

    def step3(
        param: Tuple[_CGSteihaugState, Union[float, np.ndarray]]
    ) -> _CGSteihaugState:
        iterp, z_next = param
        return iterp._replace(step=z_next, hits_boundary=False, done=True)

    # initialize the step
    p_origin = zeros_like(g)

    # init the state for the first iteration
    z = p_origin
    r = g
    d = -r
    energy = None if absdelta is None else 0.
    init_param = _CGSteihaugState(
        z=z,
        r=r,
        d=d,
        step=p_origin,
        energy=energy,
        hits_boundary=False,
        done=False,
        nit=0,
        nhev=0
    )

    # Search for the min of the approximation of the objective function.
    def body_f(iterp: _CGSteihaugState) -> _CGSteihaugState:
        z, r, d, nhev = iterp.z, iterp.r, iterp.d, iterp.nhev
        energy, nit = iterp.energy, iterp.nit

        nit += 1
        nhev += 1

        Bd = hessp_at_xk(d)
        dBd = d.dot(Bd)

        r_squared = r.dot(r)
        alpha = r_squared / dBd
        z_next = z + alpha * d

        r_next = r + alpha * Bd
        r_next_squared = r_next.dot(r_next)

        beta_next = r_next_squared / r_squared
        d_next = -r_next + beta_next * d

        accept_z_next = nit >= maxiter
        accept_z_next |= np.sqrt(r_next_squared) < resnorm
        if absdelta is None:
            energy_next = energy
        else:
            # Relative to a plain CG, `z_next` is negative
            energy_next = ((r_next + g) / 2).dot(z_next)
            energy_diff = energy - energy_next
            basically_zero = -eps * np.abs(energy)
            accept_z_next |= (energy_diff >= basically_zero
                             ) & (energy_diff < absdelta) & (nit >= miniter)

        # include a junk switch to catch the case where none should be executed
        index = np.argmax(
            np.array(
                [
                    False, dBd <= 0,
                    jft_norm(z_next, ord=norm_ord) >= trust_radius,
                    accept_z_next
                ]
            )
        )
        result = lax.switch(index, [noop, step1, step2, step3], (iterp, z_next))

        state = _CGSteihaugState(
            z=z_next,
            r=r_next,
            d=d_next,
            step=result.step,
            energy=energy_next,
            hits_boundary=result.hits_boundary,
            done=result.done,
            nhev=nhev + result.nhev,
            nit=nit
        )
        return state

    def cond_f(iterp: _CGSteihaugState) -> bool:
        return np.logical_not(iterp.done)

    # perform inner optimization to solve the constrained
    # quadratic subproblem using cg
    result = lax.while_loop(cond_f, body_f, init_param)

    pred_f = soa(result.step)
    result = _QuadSubproblemResult(
        step=result.step,
        hits_boundary=result.hits_boundary,
        pred_f=pred_f,
        nfev=0,
        njev=0,
        nhev=result.nhev + 1,
        success=True
    )

    return result
