import sys
from datetime import datetime
from functools import partial
from jax import lax
from jax import numpy as np
from jax.tree_util import (
    Partial, tree_leaves, tree_map, tree_structure, tree_reduce
)

from typing import Any, Callable, NamedTuple, Mapping, Optional, Tuple, Union

from .likelihood import doc_from
from .sugar import norm as jft_norm
from .sugar import sum_of_squares

N_RESET = 20


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
    success: Union[bool, np.ndarray]
    status: Union[int, np.ndarray]
    fun: Any
    jac: Any
    hess: Optional[np.ndarray] = None
    hess_inv: Optional[np.ndarray] = None
    nfev: Union[None, int, np.ndarray] = None
    njev: Union[None, int, np.ndarray] = None
    nhev: Union[None, int, np.ndarray] = None
    nit: Union[None, int, np.ndarray] = None
    # Trust-Region specific slots
    trust_radius: Union[None, float, np.ndarray] = None
    jac_magnitude: Union[None, float, np.ndarray] = None
    good_approximation: Union[None, bool, np.ndarray] = None


def get_dtype(v):
    if hasattr(v, "dtype"):
        return v.dtype
    else:
        return type(v)


def common_type(*trees):
    from numpy import find_common_type

    common_dtp = find_common_type(
        tuple(
            find_common_type(tuple(get_dtype(v) for v in tree_leaves(tr)), ())
            for tr in trees
        ), ()
    )
    return common_dtp


def size(tree, axis: Optional[int] = None) -> Union[int, np.ndarray]:
    if axis is not None:
        raise TypeError("axis of an arbitrary tree is ill defined")
    sizes = tree_map(np.size, tree)
    return tree_reduce(np.add, sizes)


def zeros_like(a, dtype=None, shape=None):
    return tree_map(partial(np.zeros_like, dtype=dtype, shape=shape), a)


def where(condition, x, y):
    import numpy as onp
    from itertools import repeat

    ts_c = tree_structure(condition)
    ts_x = tree_structure(x)
    ts_y = tree_structure(y)
    ts_max = (ts_c, ts_x, ts_y)[onp.argmax(
        [ts_c.num_nodes, ts_x.num_nodes, ts_y.num_nodes]
    )]

    if ts_c.num_nodes < ts_max.num_nodes:
        if ts_c.num_nodes > 1:
            raise ValueError("can not broadcast condition")
        condition = ts_max.unflatten(repeat(condition, ts_max.num_leaves))
    if ts_x.num_nodes < ts_max.num_nodes:
        if ts_x.num_nodes > 1:
            raise ValueError("can not broadcast LHS")
        x = ts_max.unflatten(repeat(x, ts_max.num_leaves))
    if ts_y.num_nodes < ts_max.num_nodes:
        if ts_y.num_nodes > 1:
            raise ValueError("can not broadcast RHS")
        y = ts_max.unflatten(repeat(y, ts_max.num_leaves))

    return tree_map(np.where, condition, x, y)


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
    for i in range(maxiter):
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
        if i % N_RESET == N_RESET - 1:
            r = mat(pos) - j
        else:
            r = r - q * alpha
        gamma = float(sum_of_squares(r))
        if time_threshold is not None and datetime.now() > time_threshold:
            info = i + 1
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
        info = i + 1
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
            i % N_RESET == N_RESET - 1, lambda x: mat(x["pos"]) - x["j"],
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


def _newton_cg(
    fun=None,
    x0=None,
    *,
    maxiter=None,
    energy_reduction_factor=0.1,
    old_fval=None,
    absdelta=None,
    norm_ord=None,
    xtol=1e-5,
    fun_and_grad=None,
    hessp=None,
    cg=cg,
    name=None,
    time_threshold=None,
    cg_kwargs=None
):
    norm_ord = 1 if norm_ord is None else norm_ord
    maxiter = 200 if maxiter is None else maxiter
    xtol = xtol * size(x0)

    pos = x0
    if fun_and_grad is None:
        from jax import value_and_grad

        fun_and_grad = value_and_grad(fun)
    cg_kwargs = {} if cg_kwargs is None else cg_kwargs

    energy, g = fun_and_grad(pos)
    if np.isnan(energy):
        raise ValueError("energy is Nan")
    status = -1
    i = 0
    for i in range(maxiter):
        cg_name = name + "CG" if name is not None else None
        # Newton approximates the potential up to second order. The CG energy
        # (`0.5 * x.T @ A @ x - x.T @ b`) and the approximation to the true
        # potential in Newton thus live on comparable energy scales. Hence, the
        # energy in a Newton minimization can be used to set the CG energy
        # convergence criterion.
        if old_fval and energy_reduction_factor:
            cg_absdelta = energy_reduction_factor * (old_fval - energy)
        else:
            cg_absdelta = None if absdelta is None else absdelta / 100.
        mag_g = jft_norm(g, ord=cg_kwargs.get("norm_ord", 1), ravel=True)
        cg_resnorm = np.minimum(0.5, np.sqrt(mag_g)) * mag_g  # taken from SciPy
        default_kwargs = {
            "absdelta": cg_absdelta,
            "resnorm": cg_resnorm,
            "norm_ord": 1,
            "name": cg_name,
            "time_threshold": time_threshold
        }
        nat_g, info = cg(Partial(hessp, pos), g, **(default_kwargs | cg_kwargs))
        if info is not None and info < 0:
            raise ValueError("conjugate gradient failed")

        naive_ls_it = 0
        dd = nat_g  # negative descent direction
        grad_scaling = 1.
        for naive_ls_it in range(9):
            new_pos = pos - grad_scaling * dd
            new_energy, new_g = fun_and_grad(new_pos)
            if new_energy <= energy:
                break

            grad_scaling /= 2
            if naive_ls_it == 5:
                if name is not None:
                    msg = f"{name}: long line search, resetting"
                    print(msg, file=sys.stderr)
                gam = float(sum_of_squares(g))
                curv = float(g.dot(hessp(pos, g)))
                grad_scaling = 1.
                dd = gam / curv * g
        else:
            grad_scaling = 0.
            nm = "N" if name is None else name
            msg = f"{nm}: WARNING: Energy would increase; aborting"
            print(msg, file=sys.stderr)
            status = -1
            break
        if name is not None:
            print(f"{name}: line search: {grad_scaling}", file=sys.stderr)

        if np.isnan(new_energy):
            raise ValueError("energy is NaN")
        energy_diff = energy - new_energy
        old_fval = energy
        energy = new_energy
        pos = new_pos
        g = new_g

        if name is not None:
            msg = f"{name}: Iteration {i+1} ⛰:{energy:.6e} Δ⛰:{energy_diff:.6e}"
            msg += f" 🞋:{absdelta:.6e}" if absdelta is not None else ""
            print(msg, file=sys.stderr)
        if absdelta is not None and 0. <= energy_diff < absdelta and naive_ls_it < 2:
            status = 0
            break
        if grad_scaling * jft_norm(dd, ord=norm_ord, ravel=True) <= xtol:
            status = 0
            break
        if time_threshold is not None and datetime.now() > time_threshold:
            status = i + 1
            break
    else:
        status = i + 1
        nm = "N" if name is None else name
        print(f"{nm}: Iteration Limit Reached", file=sys.stderr)
    return OptimizeResults(
        x=pos, success=True, status=status, fun=energy, jac=g, nit=i + 1
    )


# The following is code adapted from Nicholas Mancuso to work with pytrees
HessVP = Callable[[np.ndarray], np.ndarray]


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


def CGSteihaugSubproblem(
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


class _TrustRegionState(NamedTuple):
    x: Any
    converged: Union[bool, np.ndarray]
    status: Union[int, np.ndarray]
    fun: Any
    jac: Any
    nfev: Union[int, np.ndarray]
    njev: Union[int, np.ndarray]
    nhev: Union[int, np.ndarray]
    nit: Union[int, np.ndarray]
    trust_radius: Union[float, np.ndarray]
    jac_magnitude: Union[float, np.ndarray]
    good_approximation: Union[bool, np.ndarray]
    old_fval: Union[float, np.ndarray]


def _minimize_trust_ncg(
    fun=None,
    x0: np.ndarray = None,
    *,
    maxiter: Optional[int] = None,
    energy_reduction_factor=0.1,
    old_fval=np.nan,
    absdelta=None,
    norm_ord=None,
    gtol: float = 1e-4,
    max_trust_radius: Union[float, np.ndarray] = 1000.,
    initial_trust_radius: Union[float, np.ndarray] = 1.0,
    eta: Union[float, np.ndarray] = 0.15,
    subproblem=CGSteihaugSubproblem,
    jac: Optional[Callable] = None,
    hessp: Optional[Callable] = None,
    fun_and_grad: Optional[Callable] = None
) -> OptimizeResults:
    norm_ord = 2 if norm_ord is None else norm_ord
    maxiter = 200 if maxiter is None else maxiter

    if not (0 <= eta < 0.25):
        raise Exception("invalid acceptance stringency")
    if gtol < 0.:
        raise Exception("gradient tolerance must be positive")
    if max_trust_radius <= 0:
        raise Exception("max trust radius must be positive")
    if initial_trust_radius <= 0:
        raise ValueError("initial trust radius must be positive")
    if initial_trust_radius >= max_trust_radius:
        ve = "initial trust radius must be less than the max trust radius"
        raise ValueError(ve)

    if fun_and_grad is None:
        from jax import value_and_grad

        fun_and_grad = value_and_grad(fun)
    if hessp is None:
        from jax import grad, jvp

        jac = grad(fun) if jac is None else jac

        def hessp(primals, tangents):
            return jvp(jac, (primals, ), (tangents, ))[1]

    f_0, g_0 = fun_and_grad(x0)

    init_params = _TrustRegionState(
        converged=False,
        status=0,
        good_approximation=np.isfinite(jft_norm(g_0, ord=norm_ord)),
        nit=1,
        x=x0,
        fun=f_0,
        jac=g_0,
        jac_magnitude=jft_norm(g_0, ord=norm_ord),
        nfev=1,
        njev=1,
        nhev=0,
        trust_radius=initial_trust_radius,
        old_fval=old_fval
    )

    def _trust_region_body_f(params: _TrustRegionState) -> _TrustRegionState:
        x_k, g_k, g_k_mag = params.x, params.jac, params.jac_magnitude
        f_k, old_fval = params.fun, params.old_fval
        tr = params.trust_radius

        if energy_reduction_factor:
            cg_absdelta = energy_reduction_factor * (old_fval - f_k)
        else:
            cg_absdelta = None if absdelta is None else absdelta / 100.
        cg_resnorm = np.minimum(0.5, np.sqrt(g_k_mag)) * g_k_mag
        # TODO: add a internal success check for future subproblem approaches
        # that might not be solvable
        result = subproblem(
            f_k,
            g_k,
            partial(hessp, x_k),
            absdelta=cg_absdelta,
            resnorm=cg_resnorm,
            trust_radius=tr,
            norm_ord=norm_ord
        )

        pred_f_kp1 = result.pred_f
        x_kp1 = x_k + result.step
        f_kp1, g_kp1 = fun_and_grad(x_kp1)

        delta = f_k - f_kp1
        pred_delta = f_k - pred_f_kp1

        # update the trust radius according to the actual/predicted ratio
        rho = delta / pred_delta
        cur_tradius = np.where(rho < 0.25, tr * 0.25, tr)
        cur_tradius = np.where(
            (rho > 0.75) & result.hits_boundary,
            np.minimum(2. * tr, max_trust_radius), cur_tradius
        )

        # compute norm to check for convergence
        g_kp1_mag = jft_norm(g_kp1, ord=norm_ord, ravel=True)

        # if the ratio is high enough then accept the proposed step
        f_kp1, x_kp1, g_kp1, g_kp1_mag = where(
            rho > eta, (f_kp1, x_kp1, g_kp1, g_kp1_mag),
            (f_k, x_k, g_k, g_k_mag)
        )
        converged = g_kp1_mag < gtol
        if absdelta:
            energy_diff = f_kp1 - f_k
            converged |= (rho > eta) & (energy_diff >
                                        0.) & (energy_diff < absdelta)

        iter_params = _TrustRegionState(
            converged=converged,
            good_approximation=pred_delta > 0,
            nit=params.nit + 1,
            x=x_kp1,
            fun=f_kp1,
            jac=g_kp1,
            jac_magnitude=g_kp1_mag,
            nfev=params.nfev + result.nfev + 1,
            njev=params.njev + result.njev + 1,
            nhev=params.nhev + result.nhev,
            trust_radius=cur_tradius,
            status=params.status,
            old_fval=f_k
        )

        return iter_params

    def _trust_region_cond_f(params: _TrustRegionState) -> bool:
        return (
            np.logical_not(params.converged) & (params.nit < maxiter) &
            params.good_approximation
        )

    state = lax.while_loop(
        _trust_region_cond_f, _trust_region_body_f, init_params
    )
    status = np.where(
        state.converged,
        0,  # converged
        np.where(
            state.nit == maxiter,
            1,  # max iters reached
            np.where(
                state.good_approximation,
                -1,  # undefined
                2,  # poor approx
            )
        )
    )
    state = state._replace(status=status)

    return OptimizeResults(
        success=state.converged & state.good_approximation,
        nit=state.nit,
        x=state.x,
        fun=state.fun,
        jac=state.jac,
        nfev=state.nfev,
        njev=state.njev,
        nhev=state.nhev,
        jac_magnitude=state.jac_magnitude,
        trust_radius=state.trust_radius,
        status=state.status,
        good_approximation=state.good_approximation
    )


def newton_cg(*args, **kwargs):
    return _newton_cg(*args, **kwargs).x


def trust_ncg(*args, **kwargs):
    return _minimize_trust_ncg(*args, **kwargs).x


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
        return _minimize_trust_ncg(fun_with_args, x0, **options)

    raise ValueError(f"method {method} not recognized")
