import jax
import jax.numpy as jnp

def slq_gauss_radau(A, f, order, num_samples=1, key=None, deflate_eigvecs=None, fixed_endpoint=None):
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
    # 1. Handle input matrix or linear operator
    if callable(A):
        matvec = A
    else:
        matvec = lambda v: A @ v
    # Determine matrix dimension
    n = A.shape[0] if not callable(A) else (deflate_eigvecs.shape[0] if deflate_eigvecs is not None else None)
    if n is None:
        raise ValueError("Must provide matrix dimension (via A shape or deflate_eigvecs) when A is a function.")
    if key is None:
        raise ValueError("PRNG key is required for random vector generation.")
    
    # 2. Generate Rademacher random vectors (num_samples x n) with entries ±1
    rademacher_vecs = 2 * jax.random.bernoulli(key, 0.5, shape=(num_samples, n)).astype(jnp.float64) - 1.0
    
    # 3. Deflate known eigenvectors from random vectors (if provided)
    if deflate_eigvecs is not None:
        Q = jnp.array(deflate_eigvecs)  # ensure JAX array
        # Project out components along Q's columns: v <- v - Q (Q^T v)
        proj = Q @ (Q.T @ rademacher_vecs.T)  # shape (n, num_samples)
        rademacher_vecs = (rademacher_vecs.T - proj).T
    
    # 4. Normalize the random vectors (so that ||v||=1 for Lanczos)
    norms = jnp.linalg.norm(rademacher_vecs, axis=1)
    norms = jnp.where(norms == 0, 1.0, norms)         # avoid zero norm
    V0 = rademacher_vecs / norms[:, None]             # shape (num_samples, n)
    
    # 5. Lanczos algorithm for each probe vector (fixed number of steps = order)
    def lanczos_one(v1):
        alpha = jnp.zeros(order, dtype=v1.dtype)
        beta = jnp.zeros(order, dtype=v1.dtype)
        v_prev = jnp.zeros_like(v1)
        v_curr = v1
        # (Using a Python loop for clarity; for JIT, ensure `order` is static or use jax.lax.fori_loop)
        for i in range(order):
            w = matvec(v_curr)                          # w = A * v_curr
            alpha_i = jnp.dot(v_curr, w)                # α_i = v_i^T A v_i
            w = w - alpha_i * v_curr - (beta[i-1] * v_prev if i > 0 else 0)  # orthogonalize against previous two vectors
            beta_i1 = jnp.linalg.norm(w)                # β_{i+1} = ||w||
            beta_i1 = jnp.where(beta_i1 == 0, 0.0, beta_i1)  # if zero, Lanczos has converged early
            v_next = jnp.where(beta_i1 != 0, w / beta_i1, v_prev)
            alpha = alpha.at[i].set(alpha_i)
            beta  = beta.at[i].set(beta_i1)
            v_prev, v_curr = v_curr, v_next
        # Construct tridiagonal T (order x order) with alphas on diag and betas on off-diags
        T_diag    = alpha
        T_subdiag = beta[:-1]       # length order-1
        T = jnp.diag(T_diag) + jnp.diag(T_subdiag, 1) + jnp.diag(T_subdiag, -1)
        return alpha, beta, T
    
    # Vectorize Lanczos over all probe vectors
    alphas, betas, Ts = jax.vmap(lanczos_one, in_axes=0, out_axes=(0,0,0))(V0)
    
    # 6. Determine Gauss-Radau endpoint μ (fixed node for quadrature)
    if fixed_endpoint is None:
        # Estimate largest eigenvalue from Lanczos: use max Ritz value plus last beta
        eigs_T = jnp.linalg.eigvalsh(Ts)           # eigenvalues of each T (shape num_samples x order)
        lam_max_est = jnp.max(eigs_T)              # maximum approximate eigenvalue
        beta_m_max = jnp.max(betas[:, -1])         # largest β_m among runs
        mu = lam_max_est + beta_m_max             # choose μ slightly above estimated λ_max
    else:
        mu = fixed_endpoint
    mu = jnp.array(mu, dtype=Ts.dtype)
    
    # 7. Gauss-Radau quadrature: integrate using T extended with node at μ
    def quad_one(alpha, beta):
        m = alpha.shape[0]
        # Compute p_m(μ) and p_{m-1}(μ) via recurrence (for the characteristic polynomial of T)
        p_prev = jnp.array(1.0, dtype=alpha.dtype)         # p_0(μ) = 1
        p_curr = mu - alpha[0]                             # p_1(μ) = μ - α_1
        for j in range(2, m+1):
            # recurrence: p_j(μ) = (μ - α_j) * p_{j-1}(μ) - (β_{j-1}^2) * p_{j-2}(μ)
            p_next = (mu - alpha[j-1]) * p_curr - (beta[j-2] ** 2) * p_prev
            p_prev, p_curr = p_curr, p_next
        p_m = p_curr
        p_m_minus_1 = p_prev
        # Modified last diagonal: c = μ - (β_m^2 * p_{m-1}(μ) / p_m(μ))
        # (Ensures μ is an eigenvalue of the extended (m+1)x(m+1) matrix)
        c = jnp.where(p_m != 0, mu - (beta[m-1] ** 2) * p_m_minus_1 / p_m, mu)
        # Form extended tridiagonal matrix T_ext of size (m+1)
        T_ext = jnp.pad(jnp.diag(alpha), [(0,1),(0,1)], mode='constant')
        T_ext = T_ext.at[m-1, m].set(beta[m-1])   # last subdiag element
        T_ext = T_ext.at[m,   m-1].set(beta[m-1])
        T_ext = T_ext.at[m,   m].set(c)
        # Compute Gauss-Radau quadrature: e1^T f(T_ext) e1 = sum_k f(λ_k) * (e1·v_k)^2
        evals, evecs = jnp.linalg.eigh(T_ext)
        f_evals = f(evals)
        w_weights = evecs[0, :] ** 2              # (e1 · eigenvector_k)^2
        return jnp.dot(w_weights, f_evals)
    
    # Vectorize quadrature over all samples
    quad_vals = jax.vmap(quad_one, in_axes=(0,0))(alphas, betas)
    
    # 8. Average over samples and scale by n to get trace(f(A))
    trace_estimate = jnp.mean(quad_vals) * n
    return trace_estimate