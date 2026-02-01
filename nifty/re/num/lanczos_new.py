# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray
Matvec = Callable[[Array], Array]


def _apply_f_safely(
    f: Callable[[Array], Array],
    x: Array,
    *,
    clip_eigs: bool,
    eig_clip: float,
    clip_eigs_max: Optional[float],
    nan_to_num: bool,
) -> Array:
    """Apply a scalar function elementwise with optional clipping/sanitization."""
    if clip_eigs:
        x = jnp.clip(x, a_min=jnp.asarray(eig_clip, dtype=x.dtype))
        if clip_eigs_max is not None:
            x = jnp.clip(x, a_max=jnp.asarray(clip_eigs_max, dtype=x.dtype))
    y = f(x)
    if nan_to_num:
        y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _solve_tridiag_thomas(
    dl: Array, d: Array, du: Array, b: Array, *, diag_shift: Array
) -> Array:
    """
    Solve a tridiagonal system using the Thomas algorithm (no pivoting).

    Solves (T + diag_shift*I) x = b where:
      - dl: subdiagonal, shape (m-1,)
      - d:  diagonal,   shape (m,)
      - du: superdiag,  shape (m-1,)
      - b:  RHS,        shape (m,)

    Returns:
      x: solution vector, shape (m,)

    Notes:
      - Assumes the system is non-singular.
      - diag_shift is a scalar (possibly 0) used as a tiny stabilizer.
    """
    m = d.shape[0]
    d = d + diag_shift

    def fwd(i, state):
        dl_, d_, du_, b_ = state
        w = dl_[i - 1] / d_[i - 1]
        d_ = d_.at[i].set(d_[i] - w * du_[i - 1])
        b_ = b_.at[i].set(b_[i] - w * b_[i - 1])
        return dl_, d_, du_, b_

    dl2, d2, du2, b2 = lax.fori_loop(1, m, fwd, (dl, d, du, b))

    x = jnp.zeros_like(b2)
    x = x.at[m - 1].set(b2[m - 1] / d2[m - 1])

    def back(i, x_):
        j = m - 2 - i
        return x_.at[j].set((b2[j] - du2[j] * x_[j + 1]) / d2[j])

    return lax.fori_loop(0, m - 1, back, x)


def _dense_tridiag_from_diagonals(alpha: Array, off: Array) -> Array:
    """Build dense symmetric tridiagonal matrix from diagonal/offdiagonal."""
    return jnp.diag(alpha) + jnp.diag(off, 1) + jnp.diag(off, -1)


def _gauss_quadrature_unit(
    alpha: Array,
    off: Array,
    f: Callable[[Array], Array],
    *,
    clip_eigs: bool,
    eig_clip: float,
    clip_eigs_max: Optional[float],
    nan_to_num: bool,
) -> Array:
    """
    Compute unit-vector Gauss quadrature value: e1^T f(T) e1,
    where T is the (order x order) Lanczos tridiagonal.
    """
    T = _dense_tridiag_from_diagonals(alpha, off)
    evals, evecs = jnp.linalg.eigh(T)  # need eigenvectors for weights
    w0 = evecs[0, :] ** 2
    fe = _apply_f_safely(
        f,
        evals,
        clip_eigs=clip_eigs,
        eig_clip=eig_clip,
        clip_eigs_max=clip_eigs_max,
        nan_to_num=nan_to_num,
    )
    return jnp.dot(w0, fe)


def _radau_quadrature_unit(
    alpha: Array,
    off: Array,
    mu: Array,
    f: Callable[[Array], Array],
    *,
    eps: float,
    clip_eigs: bool,
    eig_clip: float,
    clip_eigs_max: Optional[float],
    nan_to_num: bool,
) -> Array:
    """
    Compute unit-vector Gauss–Radau quadrature value: e1^T f(T_hat) e1,
    where T_hat is T with its last diagonal modified so that mu is a Radau node.

    Implementation:
      - Solve (T_{m-1} - mu I) x = e_{m-1} using a tridiagonal solve.
      - g = x[-1] = [(T_{m-1} - mu I)^{-1}]_{m-1,m-1}
      - alpha_last_hat = mu + beta_last^2 * g
      - Form T_hat (still tridiagonal, same off-diagonals), then dense-eigh for weights.
    """
    m = alpha.shape[0]
    if m == 1:
        fe = _apply_f_safely(
            f,
            alpha,
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
        )
        return fe[0]

    # T_{m-1} has diag alpha[:-1], off off[:-1]
    diag = alpha[:-1] - mu
    dl = off[:-1]
    du = off[:-1]
    e_last = jnp.zeros((m - 1,), dtype=alpha.dtype).at[m - 2].set(1.0)

    # tiny shift prevents rare accidental singularity if mu hits Ritz value
    diag_shift = (eps * (1.0 + jnp.abs(mu))) if (eps and eps > 0.0) else 0.0
    x = _solve_tridiag_thomas(dl, diag, du, e_last, diag_shift=diag_shift)
    g = x[-1]

    beta_last = off[-1]
    beta_last = jnp.where(beta_last > eps, beta_last, 0.0)

    alpha_last_hat = mu + (beta_last**2) * g
    alpha_hat = alpha.at[m - 1].set(alpha_last_hat)

    T_hat = _dense_tridiag_from_diagonals(alpha_hat, off)
    evals, evecs = jnp.linalg.eigh(T_hat)
    w0 = evecs[0, :] ** 2
    fe = _apply_f_safely(
        f,
        evals,
        clip_eigs=clip_eigs,
        eig_clip=eig_clip,
        clip_eigs_max=clip_eigs_max,
        nan_to_num=nan_to_num,
    )
    return jnp.dot(w0, fe)


# -----------------------------------------------------------------------------
# Lanczos core (JAX-safe)
# -----------------------------------------------------------------------------
def _lanczos_tridiag_one_probe(
    v1: Array,
    matvec: Matvec,
    *,
    order: int,
    n: int,
    eps: float,
    dtype: Any,
    reorth_mode: int,  # 0 none, 1 partial, 2 full
    reorth_k: int,
) -> Tuple[Array, Array, Array]:
    """
    Lanczos recurrence for a single starting vector v1 (assumed normalized).

    Returns:
      alpha: (order,) diagonal of T
      off:   (order-1,) off-diagonal of T
      beta_full: (order,) residual norms after each step (useful for diagnostics)
    """
    alpha = jnp.zeros((order,), dtype=dtype)
    beta_full = jnp.zeros((order,), dtype=dtype)

    v_prev = jnp.zeros((n,), dtype=dtype)
    v_curr = v1.astype(dtype)

    # reorth buffers
    if reorth_mode == 0:
        Vbuf = jnp.zeros((1, 1), dtype=dtype)
        ptr = jnp.array(0, dtype=jnp.int32)
        count = jnp.array(0, dtype=jnp.int32)
        kmax_partial = 1
    elif reorth_mode == 1:
        kmax_partial = int(max(1, reorth_k))
        Vbuf = jnp.zeros((kmax_partial, n), dtype=dtype).at[0].set(v_curr)
        ptr = jnp.array(1 % kmax_partial, dtype=jnp.int32)
        count = jnp.array(1, dtype=jnp.int32)
    else:
        kmax_partial = 1
        Vbuf = jnp.zeros((order, n), dtype=dtype).at[0].set(v_curr)
        ptr = jnp.array(1, dtype=jnp.int32)
        count = jnp.array(1, dtype=jnp.int32)

    alive = jnp.array(True)

    def orthogonalize(vecs, w):
        proj = vecs @ w
        return w - (proj[:, None] * vecs).sum(axis=0)

    def step(i, state):
        alpha_, beta_full_, v_prev_, v_curr_, Vbuf_, ptr_, count_, alive_ = state

        def do_step(st):
            alpha__, beta_full__, v_prev__, v_curr__, Vbuf__, ptr__, count__, alive__ = st

            w = matvec(v_curr__)
            a = jnp.dot(v_curr__, w)
            w = w - a * v_curr__ - jnp.where(i > 0, beta_full__[i - 1] * v_prev__, 0.0)

            # Reorthogonalization (optional)
            if reorth_mode != 0:
                if reorth_mode == 2:
                    k = jnp.minimum(count__, i + 1)
                    vecs = Vbuf__[:k, :]
                    w = orthogonalize(vecs, w)
                else:
                    k = jnp.minimum(jnp.minimum(count__, i + 1), kmax_partial)
                    idx = (ptr__ - 1 - jnp.arange(k, dtype=jnp.int32)) % kmax_partial
                    vecs = Vbuf__[idx, :]
                    w = orthogonalize(vecs, w)

            b = jnp.linalg.norm(w)
            good = b > eps
            v_next = jnp.where(good, w / b, v_curr__)

            alpha__ = alpha__.at[i].set(a)
            beta_full__ = beta_full__.at[i].set(b)

            v_prev2 = v_curr__
            v_curr2 = v_next

            def store(st2):
                Vb, p, c = st2
                if reorth_mode == 0:
                    return Vb, p, c
                if reorth_mode == 1:
                    Vb2 = Vb.at[p].set(v_curr2)
                    p2 = (p + 1) % kmax_partial
                    c2 = jnp.minimum(c + 1, kmax_partial)
                    return Vb2, p2, c2
                # full
                Vb2 = lax.cond(
                    i + 1 < order,
                    lambda vb: vb.at[i + 1].set(v_curr2),
                    lambda vb: vb,
                    Vb,
                )
                return Vb2, p + 1, jnp.minimum(c + 1, order)

            Vbuf2, ptr2, count2 = store((Vbuf__, ptr__, count__))
            alive2 = alive__ & good
            return (alpha__, beta_full__, v_prev2, v_curr2, Vbuf2, ptr2, count2, alive2)

        return lax.cond(alive_, do_step, lambda st: st, state)

    state0 = (alpha, beta_full, v_prev, v_curr, Vbuf, ptr, count, alive)
    alpha, beta_full, *_ = lax.fori_loop(0, order, step, state0)

    off = beta_full[:-1]
    return alpha, off, beta_full


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def slq_gauss_radau(
    A: Union[Array, Matvec],
    f: Callable[[Array], Array],
    order: int,
    num_samples: int = 1,
    *,
    key: Array,
    deflate_eigvecs: Optional[Array] = None,
    fixed_endpoint: Optional[float] = None,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
    # --- robustness knobs ---
    eps: float = 1e-12,
    jitter: float = 0.0,
    clip_eigs: bool = True,
    eig_clip: float = 1e-300,
    clip_eigs_max: Optional[float] = None,
    nan_to_num: bool = False,
    # orthogonality control
    reorthogonalize: str = "none",  # "none" | "partial" | "full"
    reorth_k: int = 6,
    # endpoint padding
    endpoint_pad_rel: float = 1e-6,
    endpoint_pad_abs: float = 0.0,
) -> Dict[str, Array]:
    """
    Estimate tr(f(A)) for symmetric (S)PD A using Stochastic Lanczos Quadrature (SLQ),
    with optional Gauss–Radau diagnostics.

    This computes a Hutchinson trace estimator using Rademacher probes z:
        tr(f(A)) = E[z^T f(A) z]
    and approximates each quadratic form z^T f(A) z via Lanczos + Gauss quadrature:
        z^T f(A) z ≈ ||z||^2 * e1^T f(T_m) e1
    where T_m is the m=order tridiagonal Lanczos matrix.

    Additionally, you can compute Gauss–Radau diagnostics by modifying the last
    diagonal of T_m such that a chosen endpoint µ becomes a quadrature node.

    Parameters
    ----------
    A:
        Either a JAX array (n, n) or a pure-JAX matvec(v)->A@v callable.
        A should be symmetric positive definite for log/trace-log use.
    f:
        Elementwise function applied to eigenvalues (e.g. jnp.log).
        If you plan to `jax.jit` this function, mark `f` as static.
    order:
        Lanczos order (m). Higher improves quadrature accuracy but costs more matvecs.
    num_samples:
        Number of Hutchinson probe vectors.
    key:
        PRNG key used to generate Rademacher probes.
    deflate_eigvecs:
        Optional (n, p) matrix of orthonormal vectors to project out of the probes:
            z <- z - Q(Q^T z)
        This reduces variance if you already know “important” directions.
        Required when A is a callable (to infer n) unless you wrap n elsewhere.
    fixed_endpoint:
        If given, compute one-endpoint Radau diagnostic using µ=fixed_endpoint.
        Must satisfy µ >= λ_max(A) for typical “upper/lower bound” intuition.
        Mutually exclusive with (lam_min, lam_max).
    lam_min, lam_max:
        If both provided, compute two Radau diagnostics at µ=lam_min and µ=lam_max.
        The returned interval is a *diagnostic* “quadrature width”, not a strict bound
        unless assumptions on f and spectral containment are met.
        Mutually exclusive with fixed_endpoint.
    eps:
        Breakdown threshold for Lanczos. If ||w|| <= eps we stop updating (“sticky breakdown”).
        Also used as tiny stabilizer in the Radau linear solve.
    jitter:
        Adds jitter * I in the matvec: (A + jitter I) v. Default off.
    clip_eigs, eig_clip, clip_eigs_max:
        Control eigenvalue clipping before applying f (useful for log near 0).
        Set clip_eigs=False to disable.
    nan_to_num:
        Replace nan/inf in f(evals) by 0. Default off.
    reorthogonalize:
        "none"  : fastest, may lose orthogonality for large order/ill-conditioned A
        "partial": reorthogonalize against last `reorth_k` vectors (ring buffer)
        "full"  : reorthogonalize against all previous vectors (more stable, slower)
    reorth_k:
        Window size for partial reorthogonalization.
    endpoint_pad_rel, endpoint_pad_abs:
        When auto-choosing an endpoint (no lam_min/max, no fixed_endpoint),
        set µ := ritz_max*(1+endpoint_pad_rel) + endpoint_pad_abs.

    Returns
    -------
    out : dict[str, Array]
        Always includes:
          - "estimate": SLQ Gauss estimate of tr(f(A))
          - "stochastic_se": standard error across Hutchinson probes
          - "gauss_estimate", "gauss_se": same as above

        Additionally includes:
          - if fixed_endpoint provided or auto-endpoint used:
                "radau_estimate", "radau_se", "radau_endpoint"
          - if lam_min/lam_max provided:
                "radau_lo", "radau_hi", "quadrature_width", "lam_min", "lam_max"

    JIT Notes
    ---------
    - This implementation is JIT-friendly (uses lax loops; no Python loops inside).
    - For best results, jit with:
        static_argnames=("f","order","num_samples","reorthogonalize","clip_eigs","nan_to_num")
      since `f` is a Python callable and control-flow depends on several flags.
    - `A` can be a JAX array or a pure JAX matvec callable.

    Practical Notes
    ---------------
    - The dominant cost for large problems is `order * num_samples * matvec(A)`.
      The dense `eigh` is only on (order x order), typically negligible.
    """
    if fixed_endpoint is not None and (lam_min is not None or lam_max is not None):
        raise ValueError("Use either fixed_endpoint OR (lam_min, lam_max), not both.")
    if (lam_min is None) ^ (lam_max is None):
        raise ValueError("Provide both lam_min and lam_max, or neither.")
    if reorthogonalize not in ("none", "partial", "full"):
        raise ValueError("reorthogonalize must be 'none', 'partial', or 'full'.")

    reorth_mode = {"none": 0, "partial": 1, "full": 2}[reorthogonalize]

    # --- matvec & dimension ---
    if callable(A):
        if deflate_eigvecs is None:
            raise ValueError("If A is callable, provide deflate_eigvecs (or add explicit n).")
        matvec_base = A
        n = int(deflate_eigvecs.shape[0])
    else:
        A = jnp.asarray(A)
        matvec_base = lambda v: A @ v
        n = int(A.shape[0])

    # Use float64 for trace-log stability (your MGVI use case)
    dtype = jnp.float64

    if jitter != 0.0:
        matvec = lambda v: matvec_base(v).astype(dtype) + jnp.asarray(jitter, dtype=dtype) * v
    else:
        matvec = lambda v: matvec_base(v).astype(dtype)

    # --- probes (Rademacher) ---
    z = 2.0 * jax.random.bernoulli(key, 0.5, shape=(num_samples, n)).astype(dtype) - 1.0

    if deflate_eigvecs is not None:
        Q = jnp.asarray(deflate_eigvecs, dtype=dtype)
        z = z - (Q @ (Q.T @ z.T)).T

    norm2 = jnp.sum(z * z, axis=1)
    denom = jnp.where(norm2 > eps, norm2, 1.0)
    norm = jnp.sqrt(denom)
    v0 = z / norm[:, None]

    # --- Lanczos for all probes ---
    def one_probe(v1):
        return _lanczos_tridiag_one_probe(
            v1,
            matvec,
            order=order,
            n=n,
            eps=eps,
            dtype=dtype,
            reorth_mode=reorth_mode,
            reorth_k=reorth_k,
        )

    alphas, offs, _ = jax.vmap(one_probe)(v0)

    # --- Gauss SLQ estimate ---
    def gauss_one(alpha, off):
        return _gauss_quadrature_unit(
            alpha,
            off,
            f,
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
        )

    gauss_vals = jax.vmap(gauss_one)(alphas, offs)
    gauss = gauss_vals * norm2

    gauss_mean = jnp.mean(gauss)
    gauss_se = jnp.std(gauss, ddof=1) / jnp.sqrt(num_samples)

    out: Dict[str, Array] = {
        "estimate": gauss_mean,
        "stochastic_se": gauss_se,
        "gauss_estimate": gauss_mean,
        "gauss_se": gauss_se,
    }

    # --- Radau diagnostics ---
    def radau_one(alpha, off, mu):
        return _radau_quadrature_unit(
            alpha,
            off,
            mu,
            f,
            eps=eps,
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
        )

    # One-endpoint Radau (fixed_endpoint or auto)
    if fixed_endpoint is not None or (lam_min is None and lam_max is None):
        if fixed_endpoint is None:
            # Use Ritz max (from small dense T) as heuristic µ >= λ_max
            def ritz_max(alpha, off):
                T = _dense_tridiag_from_diagonals(alpha, off)
                return jnp.max(jnp.linalg.eigvalsh(T))

            mu = jnp.max(jax.vmap(ritz_max)(alphas, offs))
            mu = mu * (1.0 + endpoint_pad_rel) + endpoint_pad_abs
        else:
            mu = jnp.asarray(fixed_endpoint, dtype=dtype)

        radau_vals = jax.vmap(lambda a, o: radau_one(a, o, mu))(alphas, offs)
        radau = radau_vals * norm2
        out["radau_estimate"] = jnp.mean(radau)
        out["radau_se"] = jnp.std(radau, ddof=1) / jnp.sqrt(num_samples)
        out["radau_endpoint"] = mu

    # Two-endpoint Radau width
    if lam_min is not None and lam_max is not None:
        mu_lo = jnp.asarray(lam_min, dtype=dtype)
        mu_hi = jnp.asarray(lam_max, dtype=dtype)

        lo_vals = jax.vmap(lambda a, o: radau_one(a, o, mu_lo))(alphas, offs)
        hi_vals = jax.vmap(lambda a, o: radau_one(a, o, mu_hi))(alphas, offs)

        lo = lo_vals * norm2
        hi = hi_vals * norm2

        out["radau_lo"] = jnp.mean(lo)
        out["radau_hi"] = jnp.mean(hi)
        out["quadrature_width"] = jnp.abs(out["radau_hi"] - out["radau_lo"])
        out["lam_min"] = mu_lo
        out["lam_max"] = mu_hi

    return out


# Suggested JIT wrapper:
# slq_jit = jax.jit(
#     slq_gauss_radau,
#     static_argnames=("f", "order", "num_samples", "reorthogonalize", "clip_eigs", "nan_to_num"),
# )