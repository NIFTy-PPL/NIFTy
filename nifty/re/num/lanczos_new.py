# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

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
    """Apply scalar function elementwise with optional clipping/sanitization."""
    if clip_eigs:
        x = jnp.clip(x, a_min=jnp.asarray(eig_clip, dtype=x.dtype))
        if clip_eigs_max is not None:
            x = jnp.clip(x, a_max=jnp.asarray(clip_eigs_max, dtype=x.dtype))
    y = f(x)
    if nan_to_num:
        y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _dense_tridiag(alpha: Array, off: Array) -> Array:
    """Dense symmetric tridiagonal from diagonal/off-diagonal."""
    return jnp.diag(alpha) + jnp.diag(off, 1) + jnp.diag(off, -1)


def _solve_tridiag_thomas(
    dl: Array, d: Array, du: Array, b: Array, *, diag_shift: Array
) -> Array:
    """
    Solve (T + diag_shift*I) x = b for tridiagonal T using Thomas algorithm (no pivoting).
    Shapes:
      dl: (m-1,), d: (m,), du: (m-1,), b: (m,)
    """
    m = d.shape[0]
    d = d + diag_shift
    piv_eps = jnp.asarray(1e-30, d.dtype)

    def fwd(i, state):
        dl_, d_, du_, b_ = state
        # Guard against tiny pivots (rare for SPD-ish shifted systems, but helps nan-robustness)
        piv = jnp.where(jnp.abs(d_[i - 1]) > piv_eps, d_[i - 1], piv_eps)
        w = dl_[i - 1] / piv
        d_ = d_.at[i].set(d_[i] - w * du_[i - 1])
        b_ = b_.at[i].set(b_[i] - w * b_[i - 1])
        return dl_, d_, du_, b_

    dl2, d2, du2, b2 = lax.fori_loop(1, m, fwd, (dl, d, du, b))

    x = jnp.zeros_like(b2)
    piv_last = jnp.where(jnp.abs(d2[m - 1]) > piv_eps, d2[m - 1], piv_eps)
    x = x.at[m - 1].set(b2[m - 1] / piv_last)

    def back(i, x_):
        j = m - 2 - i
        piv = jnp.where(jnp.abs(d2[j]) > piv_eps, d2[j], piv_eps)
        return x_.at[j].set((b2[j] - du2[j] * x_[j + 1]) / piv)

    return lax.fori_loop(0, m - 1, back, x)


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
) -> Array:
    """Compute e1^T f(T) e1 for symmetric tridiagonal T with diag=alpha, offdiag=off."""
    T = _dense_tridiag(alpha, off)
    evals, evecs = jnp.linalg.eigh(T)
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

    Uses a tridiagonal solve to compute the last diagonal modification.
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

    # g = [(T_{m-1} - mu I)^{-1}]_{m-1,m-1}
    diag = alpha[:-1] - mu
    dl = off[:-1]
    du = off[:-1]
    e_last = jnp.zeros((m - 1,), dtype=alpha.dtype).at[m - 2].set(1.0)

    diag_shift = (eps * (1.0 + jnp.abs(mu))) if (eps and eps > 0.0) else jnp.asarray(0.0, alpha.dtype)
    x = _solve_tridiag_thomas(dl, diag, du, e_last, diag_shift=diag_shift)
    g = x[-1]

    beta_last = off[-1]
    beta_last = jnp.where(beta_last > eps, beta_last, 0.0)

    alpha_last_hat = mu + (beta_last**2) * g
    alpha_hat = alpha.at[m - 1].set(alpha_last_hat)

    T_hat = _dense_tridiag(alpha_hat, off)
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
# Lanczos (single probe)
# -----------------------------------------------------------------------------
def _lanczos_tridiag_one(
    v1: Array,
    matvec: Matvec,
    *,
    order: int,
    n: int,
    eps: float,
    dtype: jnp.dtype,
    reorth_mode: int,  # 0 none, 1 partial, 2 full
    reorth_k: int,
) -> Tuple[Array, Array, Array]:
    """
    Lanczos recurrence for one normalized starting vector v1.
    Returns:
      alpha: (order,) diagonal of T
      off:   (order-1,) off-diagonal of T
      beta_full: (order,) residual norms after each step (diagnostics)
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

            # optional reorth
            if reorth_mode != 0:
                if reorth_mode == 2:
                    k = jnp.minimum(count__, i + 1)
                    w = orthogonalize(Vbuf__[:k, :], w)
                else:
                    k = jnp.minimum(jnp.minimum(count__, i + 1), kmax_partial)
                    idx = (ptr__ - 1 - jnp.arange(k, dtype=jnp.int32)) % kmax_partial
                    w = orthogonalize(Vbuf__[idx, :], w)

            b = jnp.linalg.norm(w)
            good = b > eps
            b = jnp.where(good, b, 0.0)
            denom = jnp.where(good, b, 1.0)
            v_next = jnp.where(good, w / denom, v_curr__)

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
) -> Array:
    """Compute e1^T f(T) e1 for multiple scalar functions."""
    T = _dense_tridiag(alpha, off)
    evals, evecs = jnp.linalg.eigh(T)
    w0 = evecs[0, :] ** 2

    def apply_fn(fn):
        fe = _apply_f_safely(
            fn,
            evals,
            clip_eigs=clip_eigs,
            eig_clip=eig_clip,
            clip_eigs_max=clip_eigs_max,
            nan_to_num=nan_to_num,
        )
        return jnp.dot(w0, fe)

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
# Main API
# -----------------------------------------------------------------------------
def slq_gauss_radau(
    A: Union[Array, Matvec],
    f: Callable[[Array], Array],
    order: int,
    num_samples: int = 1,
    *,
    key: Array,
    n: Optional[int] = None,
    deflate_eigvecs: Optional[Array] = None,
    fixed_endpoint: Optional[float] = None,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
    extra_fns: Optional[Dict[str, Callable[[Array], Array]]] = None,
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
    # endpoint padding (auto endpoint)
    endpoint_pad_rel: float = 1e-6,
    endpoint_pad_abs: float = 0.0,
    # micro-batching of probes
    probe_batch_size: Optional[int] = None,
    # auto-endpoint strategy
    auto_endpoint_two_pass: bool = True,
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
        Number of Hutchinson probe vectors (Rademacher).
    key:
        PRNGKey.
    extra_fns:
        Optional dict of additional scalar functions evaluated with the same
        Lanczos tridiagonals. These are reported via
        `extra_{name}_estimate` and `extra_{name}_se` in the output and use
        Gauss quadrature only (no Radau diagnostics).

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
    - If fixed_endpoint is provided:
        "radau_estimate", "radau_se", "radau_endpoint"
    - If lam_min and lam_max are both provided:
        "radau_lo", "radau_hi", "quadrature_width"
    - Else (auto endpoint):
        "radau_estimate", "radau_se", "radau_endpoint"
        The endpoint is chosen from the maximum Ritz value across probes,
        padded by endpoint_pad_rel/endpoint_pad_abs.
        * auto_endpoint_two_pass=True: recompute Lanczos to evaluate Radau at
          that fixed endpoint (accurate, ~2x Lanczos cost).
        * auto_endpoint_two_pass=False: compute Radau during pass 1 using a
          running endpoint (single pass; faster but approximate).
          For a fixed endpoint, pass fixed_endpoint or lam_max.

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
    if fixed_endpoint is not None and (lam_min is not None or lam_max is not None):
        raise ValueError("Use either fixed_endpoint OR (lam_min, lam_max), not both.")
    if (lam_min is None) ^ (lam_max is None):
        raise ValueError("Provide both lam_min and lam_max, or neither.")
    if reorthogonalize not in ("none", "partial", "full"):
        raise ValueError("reorthogonalize must be 'none', 'partial', or 'full'.")
    if order < 1:
        raise ValueError("order must be >= 1.")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    reorth_mode = {"none": 0, "partial": 1, "full": 2}[reorthogonalize]
    dtype = jnp.float64

    extra_names: Tuple[str, ...] = ()
    extra_fns_tuple: Tuple[Callable[[Array], Array], ...] = ()
    if extra_fns is not None:
        if not isinstance(extra_fns, dict):
            raise ValueError("extra_fns must be a dict of name -> callable.")
        if extra_fns:
            extra_items = tuple(extra_fns.items())
            extra_names = tuple(k for k, _ in extra_items)
            extra_fns_tuple = tuple(v for _, v in extra_items)

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
            # v shape (n,) or (B,n)
            y = jax.vmap(matvec_base)(v) if v.ndim == 2 else matvec_base(v)
            y = y.astype(dtype)
            if jitter != 0.0:
                y = y + jnp.asarray(jitter, dtype=dtype) * v
            return y

        default_B = 8
    else:
        A = jnp.asarray(A, dtype=dtype)
        n = int(A.shape[0])

        def matvec(v: Array) -> Array:
            # For symmetric A, (B,n)@A equals A@(B,n)^T transposed, but this is batch-friendly.
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
    Q = None
    if deflate_eigvecs is not None:
        Q = jnp.asarray(deflate_eigvecs, dtype=dtype)

    # --- decide which diagnostics are needed ---
    need_one_endpoint = (fixed_endpoint is not None) or (lam_min is None and lam_max is None)
    need_two_endpoint = (lam_min is not None and lam_max is not None)

    # --- per-probe lanczos function ---
    def one_probe(v1):
        return _lanczos_tridiag_one(
            v1,
            matvec,
            order=order,
            n=n,
            eps=eps,
            dtype=dtype,
            reorth_mode=reorth_mode,
            reorth_k=reorth_k,
        )

    # --- per-probe quadrature functions (unit) ---
    if extra_fns_tuple:
        fns_all = (f,) + extra_fns_tuple

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
    else:

        def gauss_one(alpha, off):
            return _gauss_unit(
                alpha,
                off,
                f,
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

    # --- helpers: generate a micro-batch of Rademacher probes, deflate, normalize ---
    def make_batch_probes(batch_key: Array, bsz: int) -> Tuple[Array, Array]:
        """
        Returns:
          v0: (bsz,n) normalized starting vectors
          norm2: (bsz,) original ||z||^2, for scaling z^T f(A) z = ||z||^2 * e1^T f(T) e1
        """
        z = 2.0 * jax.random.bernoulli(batch_key, 0.5, shape=(bsz, n)).astype(dtype) - 1.0
        if Q is not None:
            # z <- z - Q(Q^T z)
            z = z - (Q @ (Q.T @ z.T)).T

        norm2 = jnp.sum(z * z, axis=1)
        denom = jnp.where(norm2 > eps, norm2, 1.0)
        v0 = z / jnp.sqrt(denom)[:, None]
        return v0, norm2

    # --- Welford accumulators for Gauss and Radau variants ---
    if extra_fns_tuple:
        ga_mean, ga_m2, ga_n = _welford_init(dtype, shape=(1 + len(extra_fns_tuple),))
    else:
        ga_mean, ga_m2, ga_n = _welford_init(dtype)
    ra_mean, ra_m2, ra_n = _welford_init(dtype)
    lo_mean, lo_m2, lo_n = _welford_init(dtype)
    hi_mean, hi_m2, hi_n = _welford_init(dtype)

    # --- for auto endpoint: track ritz max (optionally streaming) ---
    have_auto_endpoint = need_one_endpoint and (fixed_endpoint is None) and (not need_two_endpoint)
    ritz_max_stream = jnp.asarray(-jnp.inf, dtype=dtype)

    # -------------------------------------------------------------------------
    # PASS 1: compute Gauss stats (and ritz max if auto endpoint), plus
    #         Radau stats if endpoints are known (fixed or lam_min/max) or
    #         if using the single-pass running-endpoint mode.
    # -------------------------------------------------------------------------
    num_full = (num_samples // B) * B
    num_batches = num_full // B

    # pre-split keys for batches and remainder
    keys = jax.random.split(key, num_batches + 1)  # last key for remainder split
    batch_keys = keys[:num_batches]
    rem_key = keys[num_batches]

    def batch_body(carry, batch_key):
        (ga_state, ra_state, lo_state, hi_state, ritz_max_so_far) = carry

        v0_b, norm2_b = make_batch_probes(batch_key, B)
        alphas_b, offs_b, _ = jax.vmap(one_probe)(v0_b)

        # Gauss values for this batch (scaled back to z^T f(A) z)
        g_unit = jax.vmap(gauss_one)(alphas_b, offs_b)
        if extra_fns_tuple:
            g = g_unit * norm2_b[:, None]
        else:
            g = g_unit * norm2_b

        # update Welford for gauss
        ga_state2 = _welford_merge(ga_state, _welford_from_samples(g))

        # ritz max (for auto endpoint)
        ritz_max_next = ritz_max_so_far
        if have_auto_endpoint:
            # max eigenvalue of each T (order is small, this is cheap)
            def ritz_one(a, o):
                return jnp.max(jnp.linalg.eigvalsh(_dense_tridiag(a, o)))
            ritz_b = jnp.max(jax.vmap(ritz_one)(alphas_b, offs_b))
            ritz_max_next = jnp.maximum(ritz_max_so_far, ritz_b)

        # Radau computations in pass1:
        #  - fixed_endpoint provided -> one-endpoint radau
        #  - lam_min/max provided -> two-endpoint radau
        #  - auto endpoint + one-pass -> running endpoint radau
        ra_state2 = ra_state
        lo_state2 = lo_state
        hi_state2 = hi_state

        if fixed_endpoint is not None:
            mu = jnp.asarray(fixed_endpoint, dtype=dtype)
            r_unit = jax.vmap(lambda a, o: radau_one(a, o, mu))(alphas_b, offs_b)
            r = r_unit * norm2_b
            ra_state2 = _welford_merge(ra_state, _welford_from_samples(r))

        if have_auto_endpoint and (not auto_endpoint_two_pass):
            mu = ritz_max_next * (1.0 + endpoint_pad_rel) + endpoint_pad_abs
            r_unit = jax.vmap(lambda a, o: radau_one(a, o, mu))(alphas_b, offs_b)
            r = r_unit * norm2_b
            ra_state2 = _welford_merge(ra_state2, _welford_from_samples(r))

        if need_two_endpoint:
            mu_lo = jnp.asarray(lam_min, dtype=dtype)
            mu_hi = jnp.asarray(lam_max, dtype=dtype)
            lo_unit = jax.vmap(lambda a, o: radau_one(a, o, mu_lo))(alphas_b, offs_b)
            hi_unit = jax.vmap(lambda a, o: radau_one(a, o, mu_hi))(alphas_b, offs_b)
            lo = lo_unit * norm2_b
            hi = hi_unit * norm2_b
            lo_state2 = _welford_merge(lo_state, _welford_from_samples(lo))
            hi_state2 = _welford_merge(hi_state, _welford_from_samples(hi))

        return (ga_state2, ra_state2, lo_state2, hi_state2, ritz_max_next), None

    carry0 = ((ga_mean, ga_m2, ga_n), (ra_mean, ra_m2, ra_n), (lo_mean, lo_m2, lo_n), (hi_mean, hi_m2, hi_n), ritz_max_stream)
    (ga_state, ra_state, lo_state, hi_state, ritz_max_stream), _ = lax.scan(batch_body, carry0, batch_keys)

    # remainder probes (if any) — do a small vmapped block
    rem = num_samples - num_full
    if rem > 0:
        # split remainder into its own key
        v0_r, norm2_r = make_batch_probes(rem_key, rem)
        alphas_r, offs_r, _ = jax.vmap(one_probe)(v0_r)

        g_unit_r = jax.vmap(gauss_one)(alphas_r, offs_r)
        if extra_fns_tuple:
            g_r = g_unit_r * norm2_r[:, None]
        else:
            g_r = g_unit_r * norm2_r
        ga_state = _welford_merge(ga_state, _welford_from_samples(g_r))

        if have_auto_endpoint:
            def ritz_one(a, o):
                return jnp.max(jnp.linalg.eigvalsh(_dense_tridiag(a, o)))
            ritz_r = jnp.max(jax.vmap(ritz_one)(alphas_r, offs_r))
            ritz_max_stream = jnp.maximum(ritz_max_stream, ritz_r)

        if fixed_endpoint is not None:
            mu = jnp.asarray(fixed_endpoint, dtype=dtype)
            r_r = jax.vmap(lambda a, o: radau_one(a, o, mu))(alphas_r, offs_r) * norm2_r
            ra_state = _welford_merge(ra_state, _welford_from_samples(r_r))

        if have_auto_endpoint and (not auto_endpoint_two_pass):
            mu = ritz_max_stream * (1.0 + endpoint_pad_rel) + endpoint_pad_abs
            r_r = jax.vmap(lambda a, o: radau_one(a, o, mu))(alphas_r, offs_r) * norm2_r
            ra_state = _welford_merge(ra_state, _welford_from_samples(r_r))

        if need_two_endpoint:
            mu_lo = jnp.asarray(lam_min, dtype=dtype)
            mu_hi = jnp.asarray(lam_max, dtype=dtype)
            lo_r = jax.vmap(lambda a, o: radau_one(a, o, mu_lo))(alphas_r, offs_r) * norm2_r
            hi_r = jax.vmap(lambda a, o: radau_one(a, o, mu_hi))(alphas_r, offs_r) * norm2_r
            lo_state = _welford_merge(lo_state, _welford_from_samples(lo_r))
            hi_state = _welford_merge(hi_state, _welford_from_samples(hi_r))

    # -------------------------------------------------------------------------
    # PASS 2 (optional): auto-endpoint Radau (single fixed endpoint)
    # -------------------------------------------------------------------------
    mu_auto = None
    if have_auto_endpoint:
        # Choose endpoint from max Ritz value, padded.
        mu_auto = ritz_max_stream * (1.0 + endpoint_pad_rel) + endpoint_pad_abs

        if auto_endpoint_two_pass:
            # second pass: recompute Lanczos for probes, but only radau updates
            # (still streaming probes; no probe storage)
            ra_mean2, ra_m2_2, ra_n2 = _welford_init(dtype)

            keys2 = jax.random.split(jax.random.fold_in(key, 1), num_batches + 1)
            batch_keys2 = keys2[:num_batches]
            rem_key2 = keys2[num_batches]

            def batch_body2(carry, batch_key):
                (rm, rm2, rn) = carry
                v0_b, norm2_b = make_batch_probes(batch_key, B)
                alphas_b, offs_b, _ = jax.vmap(one_probe)(v0_b)
                r_b = jax.vmap(lambda a, o: radau_one(a, o, mu_auto))(alphas_b, offs_b) * norm2_b
                return _welford_merge((rm, rm2, rn), _welford_from_samples(r_b)), None

            (ra_mean2, ra_m2_2, ra_n2), _ = lax.scan(batch_body2, (ra_mean2, ra_m2_2, ra_n2), batch_keys2)

            if rem > 0:
                v0_r2, norm2_r2 = make_batch_probes(rem_key2, rem)
                alphas_r2, offs_r2, _ = jax.vmap(one_probe)(v0_r2)
                r_r2 = jax.vmap(lambda a, o: radau_one(a, o, mu_auto))(alphas_r2, offs_r2) * norm2_r2
                ra_mean2, ra_m2_2, ra_n2 = _welford_merge(
                    (ra_mean2, ra_m2_2, ra_n2), _welford_from_samples(r_r2)
                )

            ra_state = (ra_mean2, ra_m2_2, ra_n2)

    # -------------------------------------------------------------------------
    # Finalize stats + outputs
    # -------------------------------------------------------------------------
    ga_mean, ga_var, ga_n = _welford_finalize(*ga_state)
    ga_se = (
        jnp.sqrt(ga_var) / jnp.sqrt(jnp.asarray(num_samples, dtype=dtype))
        if num_samples > 1
        else jnp.asarray(0.0, dtype=dtype)
    )

    if extra_fns_tuple:
        out: Dict[str, Array] = {
            "estimate": ga_mean[0],
            "stochastic_se": ga_se[0],
            "gauss_estimate": ga_mean[0],
            "gauss_se": ga_se[0],
        }
        for idx, name in enumerate(extra_names, start=1):
            out[f"extra_{name}_estimate"] = ga_mean[idx]
            out[f"extra_{name}_se"] = ga_se[idx]
    else:
        out = {
            "estimate": ga_mean,
            "stochastic_se": ga_se,
            "gauss_estimate": ga_mean,
            "gauss_se": ga_se,
        }

    if need_one_endpoint:
        rm, rvar, rn = _welford_finalize(*ra_state)
        rse = jnp.sqrt(rvar) / jnp.sqrt(jnp.asarray(num_samples, dtype=dtype)) if num_samples > 1 else jnp.asarray(0.0, dtype=dtype)
        out["radau_estimate"] = rm
        out["radau_se"] = rse
        if fixed_endpoint is not None:
            out["radau_endpoint"] = jnp.asarray(fixed_endpoint, dtype=dtype)
        else:
            out["radau_endpoint"] = jnp.asarray(mu_auto, dtype=dtype)

    if need_two_endpoint:
        lm, lvar, ln = _welford_finalize(*lo_state)
        hm, hvar, hn = _welford_finalize(*hi_state)
        out["radau_lo"] = lm
        out["radau_hi"] = hm
        out["quadrature_width"] = jnp.abs(hm - lm)
        out["lam_min"] = jnp.asarray(lam_min, dtype=dtype)
        out["lam_max"] = jnp.asarray(lam_max, dtype=dtype)

    return out


# Suggested JIT wrapper:
# slq_jit = jax.jit(
#     slq_gauss_radau,
#     static_argnames=("f", "order", "num_samples", "reorthogonalize", "clip_eigs", "nan_to_num"),
# )
