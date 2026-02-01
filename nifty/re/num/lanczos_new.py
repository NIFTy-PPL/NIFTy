import jax
import jax.numpy as jnp

def slq_gauss_radau(A, f, order, num_samples=1, key=None, deflate_eigvecs=None, fixed_endpoint=None, lam_min=None, lam_max=None,):
    """
    Estimate trace(f(A)) for a symmetric positive-definite matrix A using Stochastic Lanczos Quadrature (SLQ) 
    with Gauss-Radau quadrature. In particular, setting f(x)=log(x) estimates log(det(A)).
    
    Parameters:
      A : Union[jax.numpy.ndarray, Callable]
          The matrix (n×n, symmetric positive-definite) or a function matvec(v) that returns A @ v for a given vector v.
      f : Callable
          Function to apply to eigenvalues of A (e.g., jnp.log for log-determinant, jnp.exp, etc.).
          It should accept a jax.numpy array and operate elementwise (vectorized).
      order : int
          Number of Lanczos iterations (quadrature order). Higher order gives higher accuracy at increased cost.
      num_samples : int, optional
          Number of random probe vectors to use (for averaging in trace estimator). Defaults to 1.
      key : jax.random.PRNGKey
          PRNG key for generating random Rademacher vectors. (Required for randomness.)
      deflate_eigvecs : jax.numpy.ndarray, optional
          Array of shape (n, p) of p orthonormal eigenvectors of A to project out from the random probes (reduces variance).
          If provided, the estimator will exactly account for those eigenmodes and randomize over the complement.
      fixed_endpoint : float, optional
          If given, use this value as the fixed endpoint μ in Gauss-Radau quadrature (should be ≥ largest eigenvalue of A).
          If None, μ is estimated automatically from Lanczos results (μ ≈ λ_max + β_m).
    
    Returns:
      trace_estimate : jax.numpy.DeviceArray
          Estimated value of tr(f(A)). For f(x)=log(x), this is an estimate of log(det(A)).
    
    The implementation uses Hutchinson's method with Rademacher vectors [oai_citation:8‡openreview.net](https://openreview.net/pdf?id=nPtmUTt8iWl#:~:text=log%20det%28K%CE%BD%29%20%3D%20tr%28log%20K%CE%BD%29,log%20K%CE%BDz%5D%2C%20%283) and Lanczos quadrature [oai_citation:9‡gerard-meurant.fr](https://gerard-meurant.fr/cgql_2012.pdf#:~:text=AVn%20%3D%20VnTn%20%E2%87%92%20f,A%29Vne1%20%3D%20v). 
    Gauss-Radau quadrature is employed with one fixed node at the spectrum's end to improve accuracy for singular functions [oai_citation:10‡openreview.net](https://openreview.net/pdf?id=nPtmUTt8iWl#:~:text=Gauss,the%20largest%20eigenvalue%20of%20K%CE%BD).
    """
    if fixed_endpoint is not None and (lam_min is not None or lam_max is not None):
        raise ValueError("Use either fixed_endpoint OR (lam_min, lam_max), not both.")
    if (lam_min is None) ^ (lam_max is None):
        raise ValueError("Provide both lam_min and lam_max, or neither.")
    if key is None:
        raise ValueError("PRNG key is required.")

    # --- matvec & dimension ---
    if callable(A):
        matvec = A
        if deflate_eigvecs is None:
            raise ValueError("If A is callable, provide deflate_eigvecs to infer n (or add an explicit n argument).")
        n = deflate_eigvecs.shape[0]
    else:
        matvec = lambda v: A @ v
        n = A.shape[0]

    # --- probes ---
    z = 2 * jax.random.bernoulli(key, 0.5, shape=(num_samples, n)).astype(jnp.float64) - 1.0

    if deflate_eigvecs is not None:
        Q = jnp.asarray(deflate_eigvecs, dtype=z.dtype)
        z = z - (Q @ (Q.T @ z.T)).T

    norm2 = jnp.sum(z * z, axis=1)
    norm = jnp.sqrt(jnp.where(norm2 == 0, 1.0, norm2))
    v0 = z / norm[:, None]

    # --- lanczos ---
    def lanczos_one(v1):
        alpha = jnp.zeros(order, dtype=v1.dtype)
        beta  = jnp.zeros(order, dtype=v1.dtype)   # beta[i] is residual norm after step i
        v_prev = jnp.zeros_like(v1)
        v_curr = v1

        for i in range(order):
            w = matvec(v_curr)
            a = jnp.dot(v_curr, w)
            w = w - a * v_curr - (beta[i - 1] * v_prev if i > 0 else 0.0)
            b = jnp.linalg.norm(w)
            v_next = jnp.where(b != 0, w / b, v_prev)

            alpha = alpha.at[i].set(a)
            beta  = beta.at[i].set(b)
            v_prev, v_curr = v_curr, v_next

        T = jnp.diag(alpha) + jnp.diag(beta[:-1], 1) + jnp.diag(beta[:-1], -1)
        return alpha, beta, T

    alphas, betas, Ts = jax.vmap(lanczos_one)(v0)

    # --- Gauss (point estimate) ---
    def gauss_unit(T):
        evals, evecs = jnp.linalg.eigh(T)
        w = evecs[0, :] ** 2
        return jnp.dot(w, f(evals))

    gauss = jax.vmap(gauss_unit)(Ts) * norm2

    gauss_mean = jnp.mean(gauss)
    gauss_se = jnp.std(gauss, ddof=1) / jnp.sqrt(num_samples)

    # --- Radau helper (unit-vector quadratic form) ---
    def radau_unit(alpha, beta, T, mu_local):
        m = alpha.shape[0]
        p_prev = jnp.array(1.0, dtype=alpha.dtype)
        p_curr = mu_local - alpha[0]
        for j in range(2, m + 1):
            p_next = (mu_local - alpha[j - 1]) * p_curr - (beta[j - 2] ** 2) * p_prev
            p_prev, p_curr = p_curr, p_next
        p_m = p_curr
        p_m_minus_1 = p_prev

        c = jnp.where(p_m != 0, mu_local - (beta[m - 1] ** 2) * p_m_minus_1 / p_m, mu_local)

        T_ext = jnp.pad(T, [(0, 1), (0, 1)], mode="constant")
        T_ext = T_ext.at[m - 1, m].set(beta[m - 1])
        T_ext = T_ext.at[m, m - 1].set(beta[m - 1])
        T_ext = T_ext.at[m, m].set(c)

        evals, evecs = jnp.linalg.eigh(T_ext)
        w = evecs[0, :] ** 2
        return jnp.dot(w, f(evals))

    out = {
        "estimate": gauss_mean,         # make Gauss the default point estimate
        "stochastic_se": gauss_se,
        "gauss_estimate": gauss_mean,
        "gauss_se": gauss_se,
    }

    # --- One-endpoint Radau (diagnostic) ---
    if fixed_endpoint is not None or (lam_min is None and lam_max is None):
        if fixed_endpoint is None:
            # heuristic endpoint ≥ λ_max: max Ritz + max residual norm
            ritz_max = jnp.max(jnp.linalg.eigvalsh(Ts))
            residual_max = jnp.max(betas[:, -1])
            mu = ritz_max + residual_max
        else:
            mu = jnp.array(fixed_endpoint, dtype=Ts.dtype)

        radau = jax.vmap(lambda a, b, T: radau_unit(a, b, T, mu))(alphas, betas, Ts) * norm2
        out["radau_estimate"] = jnp.mean(radau)
        out["radau_se"] = jnp.std(radau, ddof=1) / jnp.sqrt(num_samples)
        out["radau_endpoint"] = mu

    # --- Two-endpoint Radau width (quadrature diagnostic) ---
    if lam_min is not None and lam_max is not None:
        mu_lo = jnp.array(lam_min, dtype=Ts.dtype)
        mu_hi = jnp.array(lam_max, dtype=Ts.dtype)

        lo = jax.vmap(lambda a, b, T: radau_unit(a, b, T, mu_lo))(alphas, betas, Ts) * norm2
        hi = jax.vmap(lambda a, b, T: radau_unit(a, b, T, mu_hi))(alphas, betas, Ts) * norm2

        lo_m = jnp.mean(lo)
        hi_m = jnp.mean(hi)
        out["radau_lo"] = lo_m
        out["radau_hi"] = hi_m
        out["quadrature_width"] = jnp.abs(hi_m - lo_m)
        out["lam_min"] = lam_min
        out["lam_max"] = lam_max

    return out