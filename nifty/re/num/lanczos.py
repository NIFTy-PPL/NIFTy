# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax import random

Array = jnp.ndarray
Matvec = Callable[[Array], Array]


def lanczos_tridiag(
    mat: Callable[[jnp.ndarray], jnp.ndarray],
    v: jnp.ndarray,
    *,
    order: int,
    tol: float = 1e-12,
    # n_reortho_steps: int = 10,
):
    """Compute a Lanczos tridiagonal and its orthonormal projection matrix.

    The tridiagonal matrix is of shape (order x order) and the stack of vectors
    has shape ``(order,) + v.shape``. The output is padded with zeros after an
    early Lanczos breakdown.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    v = jnp.asarray(v)
    shape = v.shape
    n = v.size

    def flat_matvec(v_flat):
        result = jnp.asarray(mat(v_flat.reshape(shape)))
        if result.shape != shape:
            raise ValueError(
                f"shape of `mat(v)` {result.shape!r} incompatible with {shape!r}"
            )
        return result.reshape((n,))

    norm = jnp.linalg.norm(v)
    v1 = (v / norm).reshape((n,))
    alpha, off, _, basis = _lanczos_tridiag(
        v1,
        flat_matvec,
        order=order,
        eps=tol,
        reorth_mode=2,
        reorth_k=order,
        return_basis=True,
    )
    return _dense_tridiag(alpha, off), basis.reshape((order,) + shape)


def stochastic_logdet_from_lanczos(
    tridiag_stack: jnp.ndarray,
    matrix_shape0: int,
    func: Callable = jnp.log,
    *,
    tol=1e-14,
):
    """Estimate a matrix trace from a stack of Lanczos tridiagonals."""
    tridiag_stack = jnp.asarray(tridiag_stack)
    if tridiag_stack.ndim != 3 or tridiag_stack.shape[-2] != tridiag_stack.shape[-1]:
        raise ValueError("tridiag_stack must have shape (num_samples, order, order)")

    alpha = jnp.diagonal(tridiag_stack, axis1=-2, axis2=-1)
    off = jnp.diagonal(tridiag_stack, offset=1, axis1=-2, axis2=-1)
    estimates = jax.vmap(
        lambda a, o: _gauss_unit(
            a,
            o,
            func,
            clip_eigs=False,
            eig_clip=tol,
            clip_eigs_max=None,
            nan_to_num=False,
            discard_eigs_below=tol,
        )
    )(alpha, off)
    return jnp.asarray(matrix_shape0, estimates.dtype) * jnp.mean(estimates)


def stochastic_lq_logdet(
    mat: Union[jnp.ndarray, Callable],
    order: int,
    n_samples: int,
    key: Union[int, jnp.ndarray],
    *,
    shape0: Optional[int] = None,
    dtype=None,
    cmap=jax.vmap,
):
    """Estimate a log-determinant with stochastic Lanczos quadrature."""
    if not isinstance(key, jnp.ndarray):
        key = random.PRNGKey(key)

    if callable(mat) and shape0 is None:
        msg = "shape0 must be provided if `mat` is callable or has no shape attribute"
        raise ValueError(msg)
    if not callable(mat):
        mat = jnp.asarray(mat)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("mat must be a square matrix")
        if shape0 is not None and shape0 != mat.shape[0]:
            raise ValueError("shape0 does not match the matrix dimension")
        shape0 = mat.shape[0]

    result = _slq_gauss_radau(
        mat,
        jnp.log,
        order,
        n_samples,
        key=key,
        n=shape0,
        dtype=dtype,
        cmap=cmap,
        probe_batch_size=n_samples,
        reorthogonalize="full",
    )
    return result["estimate"]


def _apply_f_safely(
    f: Callable[[Array], Array],
    x: Array,
    *,
    clip_eigs: bool,
    eig_clip: float,
    clip_eigs_max: Optional[float],
    nan_to_num: bool,
) -> Array:
    """Apply scalar function elementwise with optional clipping/sanitization."""
    if clip_eigs:
        eig_floor = jnp.maximum(
            jnp.asarray(eig_clip, dtype=x.dtype),
            jnp.asarray(jnp.finfo(x.dtype).tiny, dtype=x.dtype),
        )
        x = jnp.clip(x, min=eig_floor)
        if clip_eigs_max is not None:
            x = jnp.clip(x, max=jnp.asarray(clip_eigs_max, dtype=x.dtype))
    y = f(x)
    if nan_to_num:
        y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _dense_tridiag(alpha: Array, off: Array) -> Array:
    """Dense symmetric tridiagonal from diagonal/off-diagonal."""
    return jnp.diag(alpha) + jnp.diag(off, 1) + jnp.diag(off, -1)


def _quadrature_from_eigh(
    evals: Array,
    first_evec_components: Array,
    f: Callable[[Array], Array],
    *,
    clip_eigs: bool,
    eig_clip: float,
    clip_eigs_max: Optional[float],
    nan_to_num: bool,
    discard_eigs_below: Optional[float] = None,
) -> Array:
    """Evaluate an ``e1.T @ f(T) @ e1`` quadrature from an eigendecomposition."""
    if discard_eigs_below is not None:
        threshold = jnp.asarray(discard_eigs_below, dtype=evals.dtype)
        evals = jnp.where(evals < threshold, jnp.nan, evals)
    fe = _apply_f_safely(
        f,
        evals,
        clip_eigs=clip_eigs,
        eig_clip=eig_clip,
        clip_eigs_max=clip_eigs_max,
        nan_to_num=nan_to_num,
    )
    terms = first_evec_components**2 * fe
    if discard_eigs_below is not None:
        return jnp.nansum(terms)
    return jnp.sum(terms)


# -----------------------------------------------------------------------------
# Quadrature kernels (unit-vector versions)
# -----------------------------------------------------------------------------
def _gauss_unit(
    alpha: Array,
    off: Array,
    f: Callable[[Array], Array],
    *,
    clip_eigs: bool,
    eig_clip: float,
    clip_eigs_max: Optional[float],
    nan_to_num: bool,
    discard_eigs_below: Optional[float] = None,
) -> Array:
    """Compute e1^T f(T) e1 for symmetric tridiagonal T with diag=alpha, offdiag=off."""
    return _gauss_unit_multi(
        alpha,
        off,
        (f,),
        clip_eigs=clip_eigs,
        eig_clip=eig_clip,
        clip_eigs_max=clip_eigs_max,
        nan_to_num=nan_to_num,
        discard_eigs_below=discard_eigs_below,
    )[0]


def _radau_unit(
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
    Compute e1^T f(T_hat) e1 where T_hat is the Gauss–Radau modified tridiagonal
    that forces mu to be a quadrature node.

    Uses the spectral representation of the leading tridiagonal block to
    compute the last diagonal modification.
    """
    m = alpha.shape[0]
    if m == 1:
        fe = _apply_f_safely(
            f,
            jnp.reshape(jnp.asarray(mu, dtype=alpha.dtype), (1,)),
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
        )
        return fe[0]

    beta_last = off[-1]

    def gauss_after_breakdown(_):
        return _gauss_unit(
            alpha,
            off,
            f,
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
        )

    def radau_with_extension(_):
        leading = _dense_tridiag(alpha[:-1], off[:-1])
        leading_evals, leading_evecs = jnp.linalg.eigh(leading)
        denominator = leading_evals - mu
        separation = jnp.asarray(eps, alpha.dtype) * (1.0 + jnp.abs(mu))
        separated = jnp.all(jnp.abs(denominator) > separation)
        safe_denominator = jnp.where(separated, denominator, 1.0)
        last_weights = leading_evecs[-1, :] ** 2
        g = jnp.sum(last_weights / safe_denominator)

        alpha_last_hat = mu + (beta_last**2) * g
        alpha_hat = alpha.at[m - 1].set(alpha_last_hat)
        T_hat = _dense_tridiag(alpha_hat, off)
        evals, evecs = jnp.linalg.eigh(T_hat)
        value = _quadrature_from_eigh(
            evals,
            evecs[0, :],
            f,
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
        )
        return jnp.where(separated, value, jnp.asarray(jnp.nan, alpha.dtype))

    return lax.cond(beta_last > eps, radau_with_extension, gauss_after_breakdown, None)


# -----------------------------------------------------------------------------
# Lanczos (single probe)
# -----------------------------------------------------------------------------
def _lanczos_tridiag(
    v1: Array,
    matvec: Matvec,
    *,
    order: int,
    eps: float,
    reorth_mode: int,  # 0 none, 1 partial, 2 full
    reorth_k: int,
    return_basis: bool = False,
):
    """
    Canonical Lanczos recurrence for one normalized, flat starting vector.

    Returns:
      alpha: (order,) diagonal of T
      off:   (order-1,) off-diagonal of T
      beta_full: (order,) residual norms after each step (diagnostics)
      basis: (order, n) only when return_basis=True
    """
    if return_basis and reorth_mode != 2:
        raise ValueError("return_basis requires full reorthogonalization")

    n = v1.shape[0]
    dtype = v1.dtype
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
            (
                alpha__,
                beta_full__,
                v_prev__,
                v_curr__,
                Vbuf__,
                ptr__,
                count__,
                alive__,
            ) = st

            w = matvec(v_curr__)
            a = jnp.dot(v_curr__, w)
            w = w - a * v_curr__ - jnp.where(i > 0, beta_full__[i - 1] * v_prev__, 0.0)

            # optional reorth
            if reorth_mode != 0:
                if reorth_mode == 2:
                    valid = jnp.arange(order, dtype=jnp.int32) < count__
                    vecs = jnp.where(valid[:, None], Vbuf__, 0.0)
                else:
                    slots = jnp.arange(kmax_partial, dtype=jnp.int32)
                    idx = (ptr__ - 1 - slots) % kmax_partial
                    valid = slots < count__
                    vecs = jnp.where(valid[:, None], Vbuf__[idx, :], 0.0)
                w = orthogonalize(vecs, w)

            b = jnp.linalg.norm(w)
            good = b > eps
            b = jnp.where(good, b, 0.0)
            denom = jnp.where(good, b, 1.0)
            v_next = jnp.where(good, w / denom, v_curr__)
            v_store = jnp.where(good, v_next, jnp.zeros_like(v_next))

            alpha__ = alpha__.at[i].set(a)
            beta_full__ = beta_full__.at[i].set(b)

            v_prev2 = v_curr__
            v_curr2 = v_next

            def store(st2):
                Vb, p, c = st2
                if reorth_mode == 0:
                    return Vb, p, c
                if reorth_mode == 1:
                    Vb2 = Vb.at[p].set(v_store)
                    p2 = (p + 1) % kmax_partial
                    c2 = jnp.minimum(c + 1, kmax_partial)
                    return Vb2, p2, c2
                # full
                Vb2 = lax.cond(
                    i + 1 < order,
                    lambda vb: vb.at[i + 1].set(v_store),
                    lambda vb: vb,
                    Vb,
                )
                return Vb2, p + 1, jnp.minimum(c + 1, order)

            Vbuf2, ptr2, count2 = store((Vbuf__, ptr__, count__))
            alive2 = alive__ & good
            return (alpha__, beta_full__, v_prev2, v_curr2, Vbuf2, ptr2, count2, alive2)

        return lax.cond(alive_, do_step, lambda st: st, state)

    state0 = (alpha, beta_full, v_prev, v_curr, Vbuf, ptr, count, alive)
    alpha, beta_full, _, _, Vbuf, _, _, _ = lax.fori_loop(0, order, step, state0)

    off = beta_full[:-1]
    if return_basis:
        return alpha, off, beta_full, Vbuf
    return alpha, off, beta_full


# -----------------------------------------------------------------------------
# Online statistics (Welford)
# -----------------------------------------------------------------------------
def _welford_init(dtype: jnp.dtype, shape=()):
    mean = jnp.zeros(shape, dtype=dtype)
    m2 = jnp.zeros(shape, dtype=dtype)
    count = jnp.asarray(0, dtype=jnp.int32)
    return mean, m2, count


def _welford_from_samples(x: Array):
    n = jnp.asarray(x.shape[0], dtype=jnp.int32)
    mean = jnp.mean(x, axis=0, dtype=x.dtype)
    m2 = jnp.sum((x - mean) * (x - mean), axis=0, dtype=x.dtype)
    return mean, m2, n


def _gauss_unit_multi(
    alpha: Array,
    off: Array,
    fns: Tuple[Callable[[Array], Array], ...],
    *,
    clip_eigs: bool,
    eig_clip: float,
    clip_eigs_max: Optional[float],
    nan_to_num: bool,
    discard_eigs_below: Optional[float] = None,
) -> Array:
    """Compute e1^T f(T) e1 for multiple scalar functions."""
    T = _dense_tridiag(alpha, off)
    evals, evecs = jnp.linalg.eigh(T)

    def apply_fn(fn):
        return _quadrature_from_eigh(
            evals,
            evecs[0, :],
            fn,
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
            discard_eigs_below=discard_eigs_below,
        )

    return jnp.stack([apply_fn(fn) for fn in fns])


def _welford_merge(a, b):
    # merge (mean_a, m2_a, n_a) with (mean_b, m2_b, n_b)
    mean_a, m2_a, n_a = a
    mean_b, m2_b, n_b = b
    n = n_a + n_b
    mean = jnp.where(n > 0, (n_a * mean_a + n_b * mean_b) / n, 0.0)
    delta = mean_b - mean_a
    m2 = m2_a + m2_b + delta * delta * (n_a * n_b) / jnp.where(n > 0, n, 1)
    return mean, m2, n


def _welford_finalize(mean, m2, count):
    # sample variance with ddof=1
    var = jnp.where(count > 1, m2 / (count - 1), 0.0)
    return mean, var, count


# -----------------------------------------------------------------------------
# Stochastic Lanczos quadrature
# -----------------------------------------------------------------------------
def _slq_gauss_radau(
    A: Union[Array, Matvec],
    f: Callable[[Array], Array],
    order: int,
    num_samples: int = 1,
    *,
    key: Array,
    n: Optional[int] = None,
    dtype=None,
    cmap=jax.vmap,
    deflate_eigvecs: Optional[Array] = None,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
    extra_fns: Optional[Dict[str, Callable[[Array], Array]]] = None,
    compute_radau: bool = False,
    # robustness knobs
    eps: float = 1e-12,
    jitter: float = 0.0,
    clip_eigs: bool = True,
    eig_clip: float = 1e-300,
    clip_eigs_max: Optional[float] = None,
    nan_to_num: bool = False,
    # orthogonality
    reorthogonalize: str = "none",  # "none" | "partial" | "full"
    reorth_k: int = 6,
    # micro-batching of probes
    probe_batch_size: Optional[int] = None,
) -> Dict[str, Array]:
    """
    Estimate tr(f(A)) for symmetric (S)PD A using Stochastic Lanczos Quadrature (SLQ),
    with optional Gauss–Radau diagnostics.

    Parameters
    ----------
    A:
        Either a dense SPD matrix (n,n) or a JAX-pure matvec(v)->A@v callable.
        If callable, pass `n=` or `deflate_eigvecs` (for dimension inference).
    f:
        Scalar function applied to eigenvalues (vectorized), e.g. jnp.log.
        For JIT you must pass f as a static argument.
    order:
        Lanczos steps / quadrature order (small, e.g. 20–200).
        If the Krylov space saturates early, the tridiagonal is padded with zeros.
    num_samples:
        Number of Hutchinson probe vectors (Rademacher). With one probe, the
        point estimate is returned but its stochastic standard error is NaN.
    key:
        PRNGKey.
    dtype:
        Probe and computation dtype. Defaults to JAX's current floating dtype.
    cmap:
        JAX mapping transform used for each micro-batch of probes.
    extra_fns:
        Optional dict of additional scalar functions evaluated with the same
        Lanczos tridiagonals. These are reported via
        `extra_{name}_estimate` and `extra_{name}_se` in the output and use
        Gauss quadrature only (no Radau diagnostics).
    compute_radau:
        Whether to compute two-endpoint Gauss-Radau diagnostics. This requires
        both `lam_min` and `lam_max`. If False, only the Gauss estimate and
        optional `extra_fns` are evaluated.

    Deflation
    ---------
    deflate_eigvecs:
        Optional (n,p) orthonormal eigenvectors to project out of probes:
            z <- z - Q(Q^T z)

    Gauss / Radau outputs
    ---------------------
    - "estimate" / "gauss_estimate": Gauss SLQ point estimate (mean of z^T f(A) z)
    - "stochastic_se" / "gauss_se": standard error of the Hutchinson estimator
    - For each entry in extra_fns: "extra_{name}_estimate", "extra_{name}_se"
    - If Radau diagnostics are requested:
        "radau_lo", "radau_hi", "quadrature_width"

    Robustness knobs
    ----------------
    eps:
        Breakdown guard and tiny shifts in tridiagonal solves.
    jitter:
        Adds jitter*I to A in matvec (off by default).
    clip_eigs/eig_clip/clip_eigs_max/nan_to_num:
        Protect f(eigs), especially for f=log near 0.

    Performance / memory
    --------------------
    - Probes are generated streaming in micro-batches: no (num_samples,n) allocations.
    - probe_batch_size controls memory vs speed. Larger is faster if it fits.
      If None: defaults to 32 for dense A, 8 for callable A.

    Returns
    -------
    dict of JAX arrays with estimates and diagnostics.
    """
    if (lam_min is None) ^ (lam_max is None):
        raise ValueError("Provide both lam_min and lam_max, or neither.")
    if compute_radau and lam_min is None:
        raise ValueError("compute_radau=True requires lam_min and lam_max.")
    if reorthogonalize not in ("none", "partial", "full"):
        raise ValueError("reorthogonalize must be 'none', 'partial', or 'full'.")
    if order < 1:
        raise ValueError("order must be >= 1.")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    reorth_mode = {"none": 0, "partial": 1, "full": 2}[reorthogonalize]
    dtype = jnp.asarray(0.0, dtype=dtype).dtype

    if extra_fns is not None and not isinstance(extra_fns, dict):
        raise ValueError("extra_fns must be a dict of name -> callable.")
    extra_items = tuple(extra_fns.items()) if extra_fns else ()
    extra_names = tuple(name for name, _ in extra_items)
    fns_all = (f,) + tuple(fn for _, fn in extra_items)

    # --- matvec & dimension ---
    if callable(A):
        matvec_base = A
        if n is None:
            if deflate_eigvecs is None:
                raise ValueError("If A is callable, provide n=... or deflate_eigvecs.")
            n = int(deflate_eigvecs.shape[0])
        else:
            n = int(n)

        def matvec(v: Array) -> Array:
            y = matvec_base(v).astype(dtype)
            if jitter != 0.0:
                y = y + jnp.asarray(jitter, dtype=dtype) * v
            return y

        default_B = 8
    else:
        A = jnp.asarray(A, dtype=dtype)
        n = int(A.shape[0])

        def matvec(v: Array) -> Array:
            y = jnp.matmul(v, A)
            if jitter != 0.0:
                y = y + jnp.asarray(jitter, dtype=dtype) * v
            return y

        default_B = 32

    # --- choose micro-batch size ---
    if probe_batch_size is None:
        B = min(default_B, num_samples)
    else:
        if probe_batch_size < 1:
            raise ValueError("probe_batch_size must be >= 1.")
        B = min(int(probe_batch_size), num_samples)

    # --- deflation matrix (kept in memory if provided) ---
    Q = None if deflate_eigvecs is None else jnp.asarray(deflate_eigvecs, dtype=dtype)

    mu_lo = jnp.asarray(lam_min, dtype=dtype) if compute_radau else None
    mu_hi = jnp.asarray(lam_max, dtype=dtype) if compute_radau else None

    # --- per-probe lanczos function ---
    def one_probe(v1):
        return _lanczos_tridiag(
            v1,
            matvec,
            order=order,
            eps=eps,
            reorth_mode=reorth_mode,
            reorth_k=reorth_k,
        )

    def gauss_one(alpha, off):
        return _gauss_unit_multi(
            alpha,
            off,
            fns_all,
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
        )

    def radau_one(alpha, off, mu):
        return _radau_unit(
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

    def make_batch_probes(batch_key: Array, bsz: int) -> Tuple[Array, Array]:
        """Generate, deflate, and normalize one batch of probes."""
        z = jax.random.rademacher(batch_key, shape=(bsz, n), dtype=dtype)
        if Q is not None:
            z = z - (Q @ (Q.T @ z.T)).T

        norm2 = jnp.sum(z * z, axis=1)
        denom = jnp.where(norm2 > eps, norm2, 1.0)
        v0 = z / jnp.sqrt(denom)[:, None]
        return v0, norm2

    def lanczos_batch(batch_key, bsz):
        probes, norm2 = make_batch_probes(batch_key, bsz)
        alpha, off, _ = cmap(one_probe)(probes)
        return alpha, off, norm2

    def update_radau(state, alpha, off, norm2, endpoint):
        values = cmap(lambda a, o: radau_one(a, o, endpoint))(alpha, off) * norm2
        return _welford_merge(state, _welford_from_samples(values))

    def process_first_pass(carry, batch_key, bsz):
        ga_state, lo_state, hi_state = carry
        alpha, off, norm2 = lanczos_batch(batch_key, bsz)

        gauss_values = cmap(gauss_one)(alpha, off) * norm2[:, None]
        ga_state = _welford_merge(ga_state, _welford_from_samples(gauss_values))

        if compute_radau:
            lo_state = update_radau(lo_state, alpha, off, norm2, mu_lo)
            hi_state = update_radau(hi_state, alpha, off, norm2, mu_hi)

        return ga_state, lo_state, hi_state

    num_batches, rem = divmod(num_samples, B)
    keys = jax.random.split(key, num_batches + 1)
    batch_keys = keys[:num_batches]
    rem_key = keys[num_batches]

    def batch_body(carry, batch_key):
        return process_first_pass(carry, batch_key, B), None

    empty_scalar_state = _welford_init(dtype)
    carry0 = (
        _welford_init(dtype, shape=(len(fns_all),)),
        empty_scalar_state,
        empty_scalar_state,
    )
    carry, _ = lax.scan(batch_body, carry0, batch_keys)
    if rem > 0:
        carry = process_first_pass(carry, rem_key, rem)
    ga_state, lo_state, hi_state = carry

    def mean_and_se(state):
        mean, variance, _ = _welford_finalize(*state)
        if num_samples == 1:
            return mean, jnp.full_like(variance, jnp.nan)
        return mean, jnp.sqrt(variance / num_samples)

    ga_mean, ga_se = mean_and_se(ga_state)
    out: Dict[str, Array] = {
        "estimate": ga_mean[0],
        "stochastic_se": ga_se[0],
        "gauss_estimate": ga_mean[0],
        "gauss_se": ga_se[0],
    }
    for idx, name in enumerate(extra_names, start=1):
        out[f"extra_{name}_estimate"] = ga_mean[idx]
        out[f"extra_{name}_se"] = ga_se[idx]

    if compute_radau:
        lm, _, _ = _welford_finalize(*lo_state)
        hm, _, _ = _welford_finalize(*hi_state)
        out["radau_lo"] = lm
        out["radau_hi"] = hm
        out["quadrature_width"] = jnp.abs(hm - lm)
        out["lam_min"] = mu_lo
        out["lam_max"] = mu_hi

    return out
