import jax
import jax.numpy as jnp


def slq_gauss_radau(
    A,
    f,
    order,
    num_samples=1,
    *,
    key,
    deflate_eigvecs=None,
    fixed_endpoint=None,
    lam_min=None,
    lam_max=None,
    # --- robustness knobs (all optional) ---
    eps=1e-12,                  # small guard for breakdown / clipping
    jitter=0.0,                 # add jitter * I to A in matvec; default off
    clip_eigs=True,             # clip eigenvalues before applying f
    eig_clip=1e-300,            # floor for eigenvalues when clip_eigs=True
    nan_to_num=False,           # if True: replace nan/inf in f(evals) by 0
    # orthogonality
    reorthogonalize="none",     # "none" | "partial" | "full"
    reorth_k=6,                 # only used for "partial"
    # endpoint padding (useful if you auto-pick endpoint)
    endpoint_pad_rel=1e-6,
    endpoint_pad_abs=0.0,
):
    """
    Estimate trace(f(A)) for symmetric (S)PD A using Hutchinson + Lanczos quadrature.
    - "estimate" is the Gauss/SLQ point estimate.
    - "radau_lo/hi" are computed with solve-based Gauss–Radau when lam_min/lam_max are provided.
      (Or "radau_estimate" with fixed_endpoint or auto endpoint.)
    """

    # --- input checks ---
    if fixed_endpoint is not None and (lam_min is not None or lam_max is not None):
        raise ValueError("Use either fixed_endpoint OR (lam_min, lam_max), not both.")
    if (lam_min is None) ^ (lam_max is None):
        raise ValueError("Provide both lam_min and lam_max, or neither.")
    if reorthogonalize not in ("none", "partial", "full"):
        raise ValueError("reorthogonalize must be 'none', 'partial', or 'full'.")
    if key is None:
        raise ValueError("PRNG key is required.")

    # --- matvec & dimension ---
    if callable(A):
        if deflate_eigvecs is None:
            raise ValueError("If A is callable, provide deflate_eigvecs (or add explicit n).")
        matvec_base = A
        n = deflate_eigvecs.shape[0]
    else:
        matvec_base = lambda v: A @ v
        n = A.shape[0]

    if jitter != 0.0:
        matvec = lambda v: matvec_base(v) + jitter * v
    else:
        matvec = matvec_base

    dtype = jnp.float64

    # --- safe f ---
    def apply_f(x):
        if clip_eigs:
            x = jnp.clip(x, a_min=jnp.array(eig_clip, dtype=x.dtype))
        y = f(x)
        if nan_to_num:
            y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y

    # --- probes (Rademacher) ---
    z = 2 * jax.random.bernoulli(key, 0.5, shape=(num_samples, n)).astype(dtype) - 1.0

    if deflate_eigvecs is not None:
        Q = jnp.asarray(deflate_eigvecs, dtype=dtype)
        z = z - (Q @ (Q.T @ z.T)).T

    norm2 = jnp.sum(z * z, axis=1)
    denom = jnp.where(norm2 > eps, norm2, 1.0)
    norm = jnp.sqrt(denom)
    v0 = z / norm[:, None]

    # --- Lanczos with optional reorth ---
    def lanczos_one(v1):
        alpha = jnp.zeros(order, dtype=dtype)
        beta  = jnp.zeros(order, dtype=dtype)   # beta[i] = ||w|| after step i
        v_prev = jnp.zeros_like(v1)
        v_curr = v1

        if reorthogonalize != "none":
            V = jnp.zeros((order, n), dtype=dtype).at[0].set(v_curr)
        else:
            V = None

        for i in range(order):
            w = matvec(v_curr)
            a = jnp.dot(v_curr, w)
            w = w - a * v_curr - (beta[i - 1] * v_prev if i > 0 else 0.0)

            if reorthogonalize != "none" and i > 0:
                if reorthogonalize == "full":
                    j0 = 0
                else:
                    j0 = max(0, i - reorth_k)
                for j in range(j0, i):
                    q = V[j]
                    w = w - jnp.dot(q, w) * q

            b = jnp.linalg.norm(w)
            b_safe = jnp.where(b > eps, b, 1.0)
            v_next = jnp.where(b > eps, w / b_safe, v_prev)

            alpha = alpha.at[i].set(a)
            beta  = beta.at[i].set(b)

            v_prev, v_curr = v_curr, v_next
            if reorthogonalize != "none" and i + 1 < order:
                V = V.at[i + 1].set(v_curr)

        T = jnp.diag(alpha) + jnp.diag(beta[:-1], 1) + jnp.diag(beta[:-1], -1)
        return alpha, beta, T

    alphas, betas, Ts = jax.vmap(lanczos_one)(v0)

    # --- Gauss (point estimate): z^T f(A) z ≈ ||z||^2 e1^T f(T) e1 ---
    def gauss_unit(T):
        evals, evecs = jnp.linalg.eigh(T)
        w = evecs[0, :] ** 2
        return jnp.dot(w, apply_f(evals))

    gauss_vals = jax.vmap(gauss_unit)(Ts)
    gauss = gauss_vals * norm2

    gauss_mean = jnp.mean(gauss)
    gauss_se = jnp.std(gauss, ddof=1) / jnp.sqrt(num_samples)

    out = {
        "estimate": gauss_mean,
        "stochastic_se": gauss_se,
        "gauss_estimate": gauss_mean,
        "gauss_se": gauss_se,
    }

    
    def radau_unit_solve(alpha, beta, mu):
        """
        Returns e1^T f(T_hat) e1 where T_hat is T_m with its last diagonal modified
        so that mu becomes a Radau node.

        Uses:
            alpha_hat_m = mu + beta_{m-1}^2 * [(T_{m-1} - mu I)^{-1}]_{m-1,m-1}
        where beta_{m-1} is the last off-diagonal of T_m, i.e. beta[m-2] here.
        """
        m = alpha.shape[0]
        if m == 1:
            return apply_f(alpha)[0]

        # last off-diagonal inside T_m
        beta_last = beta[m - 2]

        # Build T_{m-1} (top-left block)
        a = alpha[:-1]
        b = beta[:-2]  # off-diagonals of T_{m-1}
        Tm1 = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)

        # Solve (T_{m-1} - mu I) x = e_{m-1}  (index m-2 in 0-based)
        e_last = jnp.zeros((m - 1,), dtype=alpha.dtype).at[m - 2].set(1.0)
        M = Tm1 - mu * jnp.eye(m - 1, dtype=alpha.dtype)

        # This can still be singular if mu is exactly an eigenvalue of T_{m-1}.
        # Add a tiny diagonal shift guarded by eps to avoid NaNs in that rare case.
        if eps and eps > 0.0:
            M = M + (eps * (1.0 + jnp.abs(mu))) * jnp.eye(m - 1, dtype=alpha.dtype)

        x = jnp.linalg.solve(M, e_last)
        g = x[m - 2]  # (m-1,m-1) entry of inverse via solve with e_last

        alpha_last_hat = mu + (beta_last ** 2) * g

        # Build modified T_hat (same off-diagonals as T)
        alpha_hat = alpha.at[m - 1].set(alpha_last_hat)
        T_hat = jnp.diag(alpha_hat) + jnp.diag(beta[:-1], 1) + jnp.diag(beta[:-1], -1)

        evals, evecs = jnp.linalg.eigh(T_hat)
        w = evecs[0, :] ** 2
        return jnp.dot(w, apply_f(evals))

    # --- One-endpoint Radau (diagnostic): fixed_endpoint OR auto endpoint ---
    if fixed_endpoint is not None or (lam_min is None and lam_max is None):
        if fixed_endpoint is None:
            # auto endpoint >= max Ritz, padded
            ritz_max = jnp.max(jnp.linalg.eigvalsh(Ts))
            mu = ritz_max * (1.0 + endpoint_pad_rel) + endpoint_pad_abs
        else:
            mu = jnp.array(fixed_endpoint, dtype=Ts.dtype)

        radau_vals = jax.vmap(lambda a, b: radau_unit_solve(a, b, mu))(alphas, betas)
        radau = radau_vals * norm2
        out["radau_estimate"] = jnp.mean(radau)
        out["radau_se"] = jnp.std(radau, ddof=1) / jnp.sqrt(num_samples)
        out["radau_endpoint"] = mu

    # --- Two-endpoint Radau width (diagnostic): uses lam_min/lam_max directly ---
    if lam_min is not None and lam_max is not None:
        mu_lo = jnp.array(lam_min, dtype=Ts.dtype)
        mu_hi = jnp.array(lam_max, dtype=Ts.dtype)

        lo_vals = jax.vmap(lambda a, b: radau_unit_solve(a, b, mu_lo))(alphas, betas)
        hi_vals = jax.vmap(lambda a, b: radau_unit_solve(a, b, mu_hi))(alphas, betas)

        lo = lo_vals * norm2
        hi = hi_vals * norm2

        out["radau_lo"] = jnp.mean(lo)
        out["radau_hi"] = jnp.mean(hi)
        out["quadrature_width"] = jnp.abs(out["radau_hi"] - out["radau_lo"])
        out["lam_min"] = lam_min
        out["lam_max"] = lam_max

    return out