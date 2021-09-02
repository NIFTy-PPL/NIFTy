import sys
from datetime import datetime
from jax import numpy as np
from jax.tree_util import Partial, tree_leaves, tree_map, tree_reduce

from typing import Any, Callable, Mapping, Optional, Tuple, Union, NamedTuple

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


def size(tree, axis: Optional[int] = None):
    if axis is not None:
        raise TypeError("axis of an arbitrary tree is ill defined")
    sizes = tree_map(np.size, tree)
    return tree_reduce(np.add, sizes)


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
    norm_ord = 2 if norm_ord is None else norm_ord
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
        pos = 0. * j
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

    norm_ord = 2 if norm_ord is None else norm_ord
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
        pos = 0. * j
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
    hessp=None,
    maxiter=None,
    *,
    energy_reduction_factor=0.1,
    old_fval=None,
    absdelta=None,
    fun_and_grad=None,
    cg=cg,
    name=None,
    time_threshold=None,
    cg_kwargs=None
):
    pos = x0
    if fun_and_grad is None:
        from jax import value_and_grad

        fun_and_grad = value_and_grad(fun)
    cg_kwargs = {} if cg_kwargs is None else cg_kwargs

    energy, g = fun_and_grad(pos)
    if np.isnan(energy):
        raise ValueError("energy is Nan")
    status = -1
    for i in range(maxiter):
        cg_name = name + "CG" if name is not None else None
        # Newton approximates the potential up to second order. The CG energy
        # (`0.5 * x.T @ A @ x - x.T @ b`) and the approximation to the true
        # potential in Newton thus live on comparable energy scales. Hence, the
        # energy in a Newton minimization can be used to set the CG energy
        # convergence criterion.
        if old_fval is not None:
            cg_absdelta = energy_reduction_factor * (old_fval - energy)
        else:
            cg_absdelta = None if absdelta is None else absdelta / 100.
        mag_g = jft_norm(g, ord=cg_kwargs.get("norm_ord", 1), ravel=True)
        # SciPy scales its CG resnorm with `min(0.5, sqrt(mag_g))`
        # cg_resnorm = mag_g * np.sqrt(mag_g).clip(None, 0.5)
        cg_resnorm = mag_g / 2
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
            print(f"{nm}: WARNING: Energy would increase; aborting", file=sys.stderr)
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
        if time_threshold is not None and datetime.now() > time_threshold:
            status = i + 1
            break
    else:
        status = i + 1
        nm = "N" if name is None else name
        print(f"{nm}: Iteration Limit Reached", file=sys.stderr)
    return OptimizeResults(x=pos, success=True, status=status, fun=energy, jac=g)


def newton_cg(*args, **kwargs):
    return _newton_cg(*args, **kwargs).x


def minimize(
    fun: Optional[Callable[..., float]],
    x0,
    args: Tuple = (),
    *,
    method: str,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None
) -> OptimizeResults:
    """Minimize fun.

    NOTE, fun is assumed to actually compute fun and its gradient.
    """
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

    raise ValueError(f"method {method} not recognized")
