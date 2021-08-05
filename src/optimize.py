import sys
from datetime import datetime
from jax import numpy as np
from jax.tree_util import Partial

from typing import Any, Callable, Mapping, Optional, Tuple, Union, NamedTuple

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
    x: np.ndarray
    success: Union[bool, np.ndarray]
    status: Union[int, np.ndarray]
    fun: np.ndarray
    jac: np.ndarray
    hess: Optional[np.ndarray] = None
    hess_inv: Optional[np.ndarray] = None
    nfev: Union[None, int, np.ndarray] = None
    njev: Union[None, int, np.ndarray] = None
    nhev: Union[None, int, np.ndarray] = None
    nit: Union[None, int, np.ndarray] = None


# Taken from nifty
def cg(
    mat,
    j,
    x0=None,
    absdelta=1.,
    resnorm=None,
    norm_ord=None,
    miniter=None,
    maxiter=None,
    name=None,
    time_threshold=None
):
    norm_ord = 2 if norm_ord is None else norm_ord
    miniter = 5 if miniter is None else miniter
    maxiter = 200 if maxiter is None else maxiter

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
    # TODO(Gordian): Use `lax.while_loop`?
    for i in range(maxiter):
        if name is not None:
            print(f"{name}: Iteration {i} Energy {energy:.6e}", file=sys.stderr)
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
                msg = f"{name}: gradnorm {norm:.6e} tgt {resnorm:.6e}"
                print(msg, file=sys.stderr)
            if norm < resnorm and i > miniter:
                info = 0
                return pos, info
        new_energy = float(((r - j) / 2).dot(pos))
        if absdelta is not None:
            if name is not None:
                msg = (
                    f"{name}: ΔEnergy {energy-new_energy:.6e}"
                    f" tgt {absdelta:.6e}"
                )
                print(msg, file=sys.stderr)
            if energy - new_energy < absdelta and i > miniter:
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


def static_cg(
    mat,
    j,
    x0=None,
    absdelta=1.,
    resnorm=None,
    norm_ord=None,
    miniter=None,
    maxiter=None,
    name=None,
    **kwargs
):
    from jax.lax import cond, while_loop

    norm_ord = 2 if norm_ord is None else norm_ord
    miniter = 5 if miniter is None else miniter
    maxiter = 200 if maxiter is None else maxiter

    def continue_condition(v):
        return v["info"] < -1

    def cg_single_step(v):
        info = v["info"]
        pos, r, d, i = v["pos"], v["r"], v["d"], v["iteration"]
        previous_gamma, previous_energy = v["gamma"], v["energy"]

        i += 1

        if name is not None:
            msg = f"{name}: Iteration {v['iteration']!r} Energy {previous_energy!r}"
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

        info = np.where(gamma == 0., 0, info)
        if resnorm is not None:
            norm = jft_norm(r, ord=norm_ord, ravel=True)
            if name is not None:
                msg = f"{name}: gradnorm {norm!r} tgt {resnorm!r}"
                print(msg, file=sys.stderr)
            info = np.where((norm < resnorm) & (i > miniter), 0, info)
        # Do not compute the energy if we do not check `absdelta`
        if absdelta is not None or name is not None:
            energy = ((r - j) / 2).dot(pos)
        else:
            energy = previous_energy
        if absdelta is not None:
            if name is not None:
                msg = f"{name}: ΔEnergy {previous_energy-energy!r} tgt {absdelta!r}"
                print(msg, file=sys.stderr)
            info = np.where(
                (previous_energy - energy < absdelta) & (i > miniter), 0, info
            )
        info = np.where(i >= maxiter, i, info)

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
            cg_absdelta = np.inf if absdelta is None else absdelta / 100.
        mag_g = jft_norm(g, ord=1, ravel=True)
        # SciPy scales its CG resnorm with `min(0.5, sqrt(mag_g))`
        # cg_resnorm = mag_g * np.sqrt(mag_g).clip(None, 0.5)
        cg_resnorm = mag_g / 2
        nat_g, info = cg(
            Partial(hessp, pos),
            g,
            absdelta=cg_absdelta,
            resnorm=cg_resnorm,
            norm_ord=1,
            name=cg_name,
            time_threshold=time_threshold,
            **cg_kwargs
        )
        if info is not None and info < 0:
            raise ValueError("conjugate gradient failed")

        naive_ls_it = 0
        dd = nat_g  # negative descent direction
        grad_scaling = 1.
        new_pos = pos - grad_scaling * dd
        new_energy, new_g = fun_and_grad(new_pos)
        for naive_ls_it in range(6):
            if new_energy <= energy:
                break
            grad_scaling /= 2
            new_pos = pos - grad_scaling * dd
            new_energy, new_g = fun_and_grad(new_pos)
            if naive_ls_it == 3:
                if name is not None:
                    msg = f"{name}: long line search, resetting"
                    print(msg, file=sys.stderr)
                gam = float(sum_of_squares(g))
                curv = float(g.dot(hessp(pos, g)))
                grad_scaling = 1.
                dd = -gam / curv * g
        else:
            grad_scaling = 1.
            new_pos = pos - nat_g
            new_energy, new_g = fun_and_grad(new_pos)
            print("Warning: Energy increased", file=sys.stderr)
        if name is not None:
            print(f"{name}: line search: {grad_scaling}")

        if np.isnan(new_energy):
            raise ValueError("energy is NaN")
        energy_diff = energy - new_energy
        old_fval = energy
        energy = new_energy
        pos = new_pos
        g = new_g

        if name is not None:
            msg = (
                f"{name}: Iteration {i+1} Energy {energy:.6e}"
                f" diff {energy_diff:.6e}"
            )
            print(msg, file=sys.stderr)
        if absdelta is not None and 0. <= energy_diff < absdelta and naive_ls_it < 2:
            break
        if time_threshold is not None and datetime.now() > time_threshold:
            break
    return OptimizeResults(x=pos, success=True, status=0, fun=energy, jac=g)


def newton_cg(*args, **kwargs):
    return _newton_cg(*args, **kwargs).x


def minimize(
    fun: Optional[Callable[...,float]],
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
