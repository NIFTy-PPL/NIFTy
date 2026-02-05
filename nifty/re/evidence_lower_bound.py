#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import math
import os
from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import scipy.linalg as slg
import scipy.sparse.linalg as ssl
from jax.tree_util import tree_map

from .evi import Samples, _parse_jit
from .likelihood import Likelihood
from .logger import logger
from .num.lanczos_new import slq_gauss_radau
from .optimize_kl import _StandardHamiltonian as StandardHamiltonian
from .tree_math.vector_math import size, vdot


class _Projector(ssl.LinearOperator):
    """Computes the projector of a Matrix or LinearOperator as a LinearOperator
    given the eigenvectors of the complementary space.

    Parameters
    ----------
    eigenvectors : ndarray
        The eigenvectors representing the directions to project out.
    """

    def __init__(self, eigenvectors):
        super().__init__(eigenvectors.dtype, 2 * (eigenvectors.shape[0],))
        self.eigenvectors = eigenvectors

    def _matvec(self, x):
        V = self.eigenvectors
        # coefficients (k,)
        c = V.T.conj() @ x
        # projection
        return x - V @ c

    def _rmatvec(self, x):
        return self._matvec(x)


class _ProjectedMetric(ssl.LinearOperator):
    def __init__(self, metric, projector):
        super().__init__(dtype=metric.dtype, shape=metric.shape)
        self.metric = metric
        self.projector = projector

    def _matvec(self, x):
        px = self.projector.matvec(x)
        mpx = self.metric.matvec(px)
        return self.projector.matvec(mpx)

    def _rmatvec(self, x):
        return self._matvec(x)


def _explicify(M):
    identity = np.identity(M.shape[0], dtype=np.float64)
    return np.column_stack([M.matvec(v) for v in identity])


def _estimate_eigenvalues(metric, eigenvectors):
    eigenvalues = np.empty(eigenvectors.shape[1], dtype=np.float64)
    for i, vec in enumerate(eigenvectors.T):
        eigenvalues[i] = np.vdot(vec, metric.matvec(vec))
    return np.real_if_close(eigenvalues)


def _orthonormalize_columns(eigenvectors):
    if eigenvectors.size == 0:
        return eigenvectors
    q, _ = np.linalg.qr(eigenvectors)
    return q


def _orthonormality_error(eigenvectors, n_probes):
    if eigenvectors.size == 0:
        return 0.0
    n_vectors = eigenvectors.shape[1]
    n_probes = min(n_probes, n_vectors)
    rng = np.random.default_rng(0)
    probes = rng.standard_normal((n_vectors, n_probes))
    projected = eigenvectors.conj().T @ (eigenvectors @ probes)
    diff = projected - probes
    return float(np.max(np.abs(diff)))


def _save_eigensystem(output_directory, prefix, eigenvalues, eigenvectors, *, verbose):
    if output_directory is None:
        return
    if output_directory == "":
        output_directory = "."
    os.makedirs(output_directory, exist_ok=True)
    base = os.path.join(output_directory, prefix)
    if verbose:
        logger.info(
            f"Saving metric eigensystem to "
            f"{base}_eigenvalues.npy and {base}_eigenvectors.npy."
        )
    np.save(f"{base}_eigenvalues.npy", eigenvalues)
    if eigenvectors is not None:
        np.save(f"{base}_eigenvectors.npy", eigenvectors)


def _make_linop(matvec, shape, dtype):
    def mv(x):
        return np.asarray(matvec(jnp.asarray(x)))

    return ssl.LinearOperator(shape=shape, dtype=dtype, matvec=mv)


def _ravel_metric(metric, position, dtype, metric_jit):
    def ravel(x):
        return jax.flatten_util.ravel_pytree(x)[0]

    shp, unravel = jax.flatten_util.ravel_pytree(position)
    shape = 2 * (shp.size,)

    def met(x, *, position):
        return ravel(metric(position, unravel(x)))

    metric_jit = _parse_jit(metric_jit)
    met = partial(metric_jit(met), position=position)

    linop = _make_linop(met, shape, dtype)
    return linop, met, shp.size


def _make_data_operator(likelihood, position, dtype, metric_jit):
    if likelihood.lsm_tangents_shape is None:
        raise ValueError("lsm_tangents_shape is required for data-space projection.")

    def zeros_like_shape(s):
        return jnp.zeros(s.shape, dtype=s.dtype)

    data_zeros = tree_map(zeros_like_shape, likelihood.lsm_tangents_shape)
    data_flat, data_unravel = jax.flatten_util.ravel_pytree(data_zeros)

    def ravel_data(x):
        return jax.flatten_util.ravel_pytree(x)[0]

    def data_op(x, *, position):
        u = data_unravel(x)
        v = likelihood.left_sqrt_metric(position, u)
        w = likelihood.right_sqrt_metric(position, v)
        return ravel_data(w)

    metric_jit = _parse_jit(metric_jit)
    data_op = partial(metric_jit(data_op), position=position)

    linop = _make_linop(data_op, 2 * (data_flat.size,), dtype)
    return linop, data_op, data_flat.size


def _eigsh(
    metric,
    metric_size,
    n_eigenvalues,
    tot_dofs,
    min_lh_eval=1e-4,
    eigenvalue_shift=1.0,
    n_batches=10,
    tol=0.0,
    early_stop=True,
    verbose=True,
    output_directory=None,
    save_eigensystem_prefix="metric",
    resume_eigenvectors=None,
    resume_eigenvalues=None,
    orthonormalize_eigenvectors=True,
    orthonormalize_every_n_batches=5,
    orthonormalize_threshold=1e-6,
    orthonormalize_n_probes=2,
):
    eigenvectors = None
    eigenvalues = None
    if resume_eigenvalues is not None and resume_eigenvectors is None:
        raise ValueError("resume_eigenvalues requires resume_eigenvectors.")
    if orthonormalize_eigenvectors:
        if (
            not isinstance(orthonormalize_every_n_batches, int)
            or orthonormalize_every_n_batches < 1
        ):
            raise ValueError(
                "orthonormalize_every_n_batches must be a positive integer."
            )
        if orthonormalize_threshold is not None and orthonormalize_threshold <= 0:
            raise ValueError("orthonormalize_threshold must be positive.")
        if not isinstance(orthonormalize_n_probes, int) or orthonormalize_n_probes < 1:
            raise ValueError("orthonormalize_n_probes must be a positive integer.")
        if resume_eigenvectors is not None and resume_eigenvalues is None:
            raise ValueError(
                "resume_eigenvalues is required when orthonormalize_eigenvectors=True."
            )
    if n_eigenvalues > tot_dofs:
        raise ValueError(
            "Number of requested eigenvalues "
            "exceeds the number of relevant degrees of freedom!"
        )

    batch_counter = 0
    if resume_eigenvectors is not None:
        eigenvectors = np.asarray(resume_eigenvectors)
        if eigenvectors.ndim != 2:
            raise ValueError("resume_eigenvectors must be a 2D array.")
        if eigenvectors.shape[0] != metric_size:
            raise ValueError("resume_eigenvectors does not match the operator size.")
        if resume_eigenvalues is None:
            eigenvalues = _estimate_eigenvalues(metric, eigenvectors)
        else:
            eigenvalues = np.asarray(resume_eigenvalues)
        if eigenvalues.ndim != 1:
            raise ValueError("resume_eigenvalues must be a 1D array.")
        if eigenvalues.size != eigenvectors.shape[1]:
            raise ValueError(
                "resume_eigenvalues and resume_eigenvectors have mismatched sizes."
            )
        order = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if eigenvalues.size > n_eigenvalues:
            eigenvalues = eigenvalues[:n_eigenvalues]
            eigenvectors = eigenvectors[:, :n_eigenvalues]
        if orthonormalize_eigenvectors and eigenvectors is not None:
            error = (
                _orthonormality_error(eigenvectors, orthonormalize_n_probes)
                if orthonormalize_threshold is not None
                else None
            )
            if error is not None and error > orthonormalize_threshold:
                if verbose:
                    logger.info(
                        "Re-orthonormalizing eigenvectors ("
                        f"orthonormality error {error:.2e} > "
                        f"{orthonormalize_threshold:.2e})."
                    )
                eigenvectors = _orthonormalize_columns(eigenvectors)
        if verbose:
            logger.info(
                f"Resuming eigenvalue computation with {eigenvalues.size} "
                f"precomputed eigenvalues."
            )

    if eigenvalues is not None and eigenvalues.size > tot_dofs:
        raise ValueError(
            "Number of provided eigenvectors exceeds relevant degrees of freedom."
        )

    if (
        early_stop
        and eigenvalues is not None
        and abs(eigenvalue_shift - np.min(eigenvalues)) < min_lh_eval
    ):
        return eigenvalues, eigenvectors

    n_precomputed = 0 if eigenvalues is None else eigenvalues.size
    remaining_eigenvalues = n_eigenvalues - n_precomputed
    if remaining_eigenvalues <= 0:
        return eigenvalues, eigenvectors

    if tot_dofs == n_eigenvalues and n_precomputed == 0:
        # Compute exact eigensystem
        if verbose:
            logger.info(f"Computing all {tot_dofs} relevant eigenvalues.")
        if output_directory is None:
            eigenvalues = slg.eigh(
                _explicify(metric),
                eigvals_only=True,
                subset_by_index=[metric_size - tot_dofs, metric_size - 1],
            )
            eigenvalues = np.flip(eigenvalues)
        else:
            eigvals, eigvecs = slg.eigh(
                _explicify(metric),
                eigvals_only=False,
                subset_by_index=[metric_size - tot_dofs, metric_size - 1],
            )
            idx = np.argsort(-eigvals)
            eigenvalues = eigvals[idx]
            eigenvectors = eigvecs[:, idx]
            _save_eigensystem(
                output_directory,
                save_eigensystem_prefix,
                eigenvalues,
                eigenvectors,
                verbose=verbose,
            )
    else:
        # Set up batches based on total n_eigenvalues, then skip precomputed.
        base = n_eigenvalues // n_batches
        remainder = n_eigenvalues % n_batches
        full_batches = [base + 1] * remainder + [base] * (n_batches - remainder)
        full_batches = [batch for batch in full_batches if batch > 0]
        batches = []
        skip = n_precomputed
        for batch in full_batches:
            if skip >= batch:
                skip -= batch
                continue
            if skip > 0:
                batch -= skip
                skip = 0
            batches.append(batch)
        projected_metric = metric
        if eigenvectors is not None:
            if orthonormalize_eigenvectors:
                error = (
                    _orthonormality_error(eigenvectors, orthonormalize_n_probes)
                    if orthonormalize_threshold is not None
                    else None
                )
                if error is not None and error > orthonormalize_threshold:
                    if verbose:
                        logger.info(
                            "Re-orthonormalizing eigenvectors ("
                            f"orthonormality error {error:.2e} > "
                            f"{orthonormalize_threshold:.2e})."
                        )
                    eigenvectors = _orthonormalize_columns(eigenvectors)
            projector = _Projector(eigenvectors)
            projected_metric = _ProjectedMetric(metric, projector)

        for batch in batches:
            if verbose:
                logger.info(f"\nNumber of eigenvalues being computed: {batch}")
            # Get eigensystem for current batch
            eigvals, eigvecs = ssl.eigsh(
                projected_metric,
                k=batch,
                tol=tol,
                return_eigenvectors=True,
                which="LM",
            )
            i = np.argsort(-eigvals)
            eigvals, eigvecs = eigvals[i], eigvecs[:, i]
            eigenvalues = (
                eigvals
                if eigenvalues is None
                else np.concatenate((eigenvalues, eigvals))
            )
            eigenvectors = (
                eigvecs if eigenvectors is None else np.hstack((eigenvectors, eigvecs))
            )
            batch_counter += 1
            if orthonormalize_eigenvectors:
                error = (
                    _orthonormality_error(eigenvectors, orthonormalize_n_probes)
                    if orthonormalize_threshold is not None
                    else None
                )
                cadence = batch_counter % orthonormalize_every_n_batches == 0
                if (error is not None and error > orthonormalize_threshold) or cadence:
                    if verbose:
                        reason = (
                            f"orthonormality error {error:.2e} > "
                            f"{orthonormalize_threshold:.2e}"
                            if error is not None and error > orthonormalize_threshold
                            else f"batch cadence every {orthonormalize_every_n_batches}"
                        )
                        logger.info(f"Re-orthonormalizing eigenvectors ({reason}).")
                    eigenvectors = _orthonormalize_columns(eigenvectors)
            _save_eigensystem(
                output_directory,
                save_eigensystem_prefix,
                eigenvalues,
                eigenvectors,
                verbose=verbose,
            )
            if verbose:
                done = eigenvalues.size
                pct = math.ceil(100.0 * done / n_eigenvalues)
                logger.info(f"Eigenvalue progress: {done}/{n_eigenvalues} ({pct}%)")

            if early_stop and abs(eigenvalue_shift - np.min(eigenvalues)) < min_lh_eval:
                break
            # Project out subspace of already computed eigenvalues
            projector = _Projector(eigenvectors)
            projected_metric = _ProjectedMetric(metric, projector)
    return eigenvalues, eigenvectors


def estimate_evidence_lower_bound(
    likelihood,
    samples,
    n_eigenvalues,
    *,
    compute_all=False,
    min_lh_eval=1e-3,
    n_batches=10,
    tol=0.0,
    verbose=True,
    metric_jit=True,
    output_directory="",
    save_eigensystem_prefix="metric",
    resume_eigenvectors=None,
    resume_eigenvalues=None,
    orthonormalize_eigenvectors=True,
    orthonormalize_every_n_batches=8,
    orthonormalize_threshold=1e-6,
    orthonormalize_n_probes=2,
    trace_log_method="eigsh",
    trace_log_space="signal",
    slq_order=64,
    slq_num_samples=8,
    slq_key=None,
    slq_kwargs=None,
    use_radau_as_bound=False,
    slq_jit=False,
    analytic_prior_term=False,
):
    """Provides an estimate for the Evidence Lower Bound (ELBO).

    Statistical inference deals with the problem of hypothesis testing, given
    some data and models that can describe it. In general, it is hard to find a
    good metric to discern between different models. In Bayesian Inference, the
    Bayes factor can serve this purpose. To compute the Bayes factor it is
    necessary to calculate the evidence, given the specific model :math:`p(
    d|\\text{model})` at hand. Then, the ratio between the evidence of a model
    A and the one of a model B represents how much more likely it is that model
    A represents the data better than model B.

    The evidence for an approximated-inference problem can in principle be
    calculated. However, this is only practically feasible in a low-dimensional
    setting. What often can be computed is

    .. math ::
        \\log(p(d)) - D_\\text{KL} \\left[ Q(\\theta(\\xi)|d) || p(\\theta(\\xi) | d) \\right] = -\\langle H(\\theta(
        \\xi), d)\\rangle + \\frac1 2 \\left( N + \\text{Tr } \\log\\Lambda\\right),

    where :math:`D_\\text{KL} \\left[ Q || p \\right]` is the Kullback-Leibler
    (KL) divergence between the approximating posterior distribution :math:`Q`
    and the actual posterior :math:`p`. Since the Kullback-Leibler divergence
    :math:`D_\\text{KL} [\\cdot, \\cdot] \\geq 0` is positive definite, it is
    convenient to consider the lower bound

    .. math ::
        \\log(p(d)) \\geq -\\langle H(\\theta(\\xi), d)\\rangle + \\frac1 2 \\left(N + \\text{Tr } \\log\\Lambda
        \\right),

    which takes the name of Evidence Lower Bound (ELBO).

    If the KL divergence is well minimized (which should always be the case
    when a Variational Inference approach is followed), then it is possible to
    utilize the ELBO (as a proxy for the actual evidences) and calculate the
    Bayes factors for model comparison.

    Parameters
    ----------
    likelihood : :class:`nifty.re.likelihood.Likelihood`
        Log-likelihood of the model.
    samples : :class:`nifty.re.evi.Samples`
        Collection of samples from the posterior distribution.
    n_eigenvalues : int
        Maximum number of eigenvalues to compute exactly. If
        `trace_log_method="slq"`, the remaining trace-log term is estimated
        via stochastic Lanczos quadrature. If `trace_log_method="eigsh"`, the
        remainder is approximated using the smallest computed eigenvalue. 
        Note that if `n_eigenvalues` equals the total number
        of relevant degrees of freedom of the problem, all relevant eigenvalues
        are always computed irrespective of other stopping criteria.
    compute_all : bool
        If True, compute all eigenvalues and eigenvectors of the relevant
        metric subspace. Overrides `n_eigenvalues`.
    min_lh_eval : float
        Smallest eigenvalue of the likelihood to be considered. If the
        estimated eigenvalues become smaller than 1 + `min_lh_eval`, the
        eigenvalue estimation terminates and uses the smallest eigenvalue as a
        proxy for all remaining eigenvalues in the trace-log estimation.
        Default is 1e-3.
    n_batches : int
        Number of batches into which the eigenvalue estimation gets subdivided.
        The batch schedule is defined for the total `n_eigenvalues`; when
        resuming, the remaining work continues with the tail of this schedule.
        Only after completing one batch the early stopping criterion based on
        `min_lh_eval` is checked for.
    tol : Optional[float]
        Tolerance on the eigenvalue calculation. Zero indicates machine
        precision. Default is 0.
    verbose : Optional[bool]
        Print list of eigenvalues and summary of evidence calculation. Default
        is True.
    metric_jit : bool or callable
        Whether to jit the metric. Default is True.
    output_directory : Optional[str]
        If set, saves the cumulative metric eigenvalues and eigenvectors after
        each batch to `{output_directory}/{prefix}_eigenvalues.npy` and
        `{output_directory}/{prefix}_eigenvectors.npy`.
    save_eigensystem_prefix : str
        Prefix for eigensystem filenames. A suffix `_signal` or `_data` is
        appended automatically depending on the selected trace-log space.
        Default is "metric".
    resume_eigenvectors : Optional[np.ndarray]
        Precomputed eigenvectors to resume the eigenvalue calculation. The
        array is expected to be shaped `(metric_size, n_vectors)`.
    resume_eigenvalues : Optional[np.ndarray]
        Eigenvalues corresponding to `resume_eigenvectors`. If not provided,
        they are estimated via the metric. Required when
        `orthonormalize_eigenvectors` is True and `resume_eigenvectors` is set.
    orthonormalize_eigenvectors : bool
        If True, re-orthonormalize the cumulative eigenvector basis before
        projecting out the corresponding subspace. Default is True.
    orthonormalize_every_n_batches : int
        Re-orthonormalize every N batches when `orthonormalize_eigenvectors`
        is True. Default is 8.
    orthonormalize_threshold : Optional[float]
        Re-orthonormalize whenever a randomized probe of `V.T @ V` deviates
        from the identity by more than this threshold. Set to `None` to disable
        this check. Default is 1e-6.
    orthonormalize_n_probes : int
        Number of randomized probe vectors used to estimate the orthonormality
        error. Default is 2.
    trace_log_method : {"eigsh", "slq"}
        Strategy for estimating the trace-log term. "eigsh" uses exact
        eigenvalues plus a minimum-eigenvalue approximation for the remainder.
        "slq" uses exact eigenvalues plus SLQ Gauss–Radau for the remainder.
        When using SLQ and at least one eigenvalue is computed, the SLQ call
        is provided with bounds based on the smallest computed eigenvalue.
    trace_log_space : {"auto", "signal", "data"}
        Space in which to estimate the trace-log term. "signal" uses the full
        metric in signal space with f=log. "data" uses the implicit
        data-space operator LSM^T LSM with f=log1p. "auto" selects the smaller
        of data- and signal-space dimensions (falls back to "signal" if
        data-space shapes are unavailable). Default is "signal".
    slq_order : int
        Lanczos order for SLQ (only used when trace_log_method="slq").
    slq_num_samples : int
        Number of Hutchinson probes for SLQ (only used when
        trace_log_method="slq").
    slq_key : int or jax.Array
        PRNG seed or key for SLQ. If None, a deterministic default is used.
    slq_kwargs : Optional[dict]
        Additional kwargs forwarded to `slq_gauss_radau` (excluding A, f,
        order, num_samples, key, n, deflate_eigvecs, lam_min, lam_max, and
        fixed_endpoint, and extra_fns).
    use_radau_as_bound : bool
        If True, and SLQ returns two-endpoint Radau bounds (requires valid
        `lam_min`/`lam_max`), use the upper Radau bound on the *remaining*
        trace-log to build a more conservative `lower_error`. This is a
        heuristic and is not guaranteed to be a rigorous bound for arbitrary
        `f` (including log/log1p) unless the corresponding conditions for
        Gauss–Radau bounds are met. Default is False, in which case the SLQ
        standard error is used.
    slq_jit : bool
        If True, JIT-compile the SLQ computation. This closes over `A`, `f`,
        `order`, and `num_samples` so they are treated as static and avoids
        JAX tracer errors with callables. Useful when running multiple SLQ
        evaluations with the same shapes; for one-off runs the compile cost
        may outweigh the speedup. Default is False.
    analytic_prior_term : bool
        If True, compute the quadratic prior term
        :math:`\\frac{1}{2}\\langle \\xi^\\dagger \\xi \\rangle_q` analytically
        as :math:`\\frac{1}{2}(\\mathrm{Tr}\\,\\Sigma + \\bar\\xi^\\dagger\\bar\\xi)`
        with :math:`\\Sigma = \\Lambda^{-1}`. The trace is estimated from the
        exact eigenvalues plus (if needed) an SLQ remainder using
        :math:`f(x)=1/x` (signal space) or :math:`f(x)=1/(1+x)` (data space).
        Requires `trace_log_method="slq"` or `compute_all=True` when not all
        eigenvalues are computed. Default is False.

    Returns
    -------
    `elbo_samples` : np.array
        List of elbo samples from the posterior distribution. The samples are
        returned to allow for more accurate elbo statistics.
    stats : dict
        Dictionary with a summary of the statistics of the estimated ELBO.
        The keys of this dictionary are:

        - `elbo_mean`: mean of the ELBO samples.
        - `elbo_std`: standard deviation of the ELBO samples (sampling spread).
        - `elbo_se`: standard error of the ELBO mean (`elbo_std / sqrt(n_samples)`).
        - `elbo_up`: `elbo_mean + elbo_std` (one-sigma upper envelope).
        - `elbo_lw`: `elbo_mean - elbo_std - lower_error` (conservative lower envelope).
        - `lower_error`: estimate of the residual trace-log uncertainty.
          For `trace_log_method="eigsh"` this is the legacy lower-bound term
          based on the smallest computed eigenvalue. For `trace_log_method="slq"`
          it is based on the SLQ standard error of the remaining trace-log.
          If `analytic_prior_term=True`, an additional term from the SLQ
          standard error of the trace-inverse estimate is included.
        - `trace_inv_exact`, `trace_inv_slq`, `trace_inv_se`, `trace_inv_total`,
          `prior_mean_sq`, `prior_term`: returned when `analytic_prior_term=True`.

    Warning
    -------
    To perform Variational Inference there is no need to take into account
    quantities that are not explicitly dependent on the inferred parameters.
    Explicitly calculating these terms can be expensive, therefore they are
    neglected in NIFTy. Since in most cases they are also not required for model
    comparison, the provided estimate may not include terms which are constant
    in these parameters. Only when comparing models for which the likelihood
    includes (possibly data-dependent) constants (or when the ELBO is needed
    to approximate the true evidence) these contributions have to be considered.
    For example, for a Gaussian distributed signal and a linear problem
    (Wiener Filter problem) the only term missing is
    :math:`-\\frac1 2 \\log \\det |2 \\pi N|`,
    where :math:`N` is the noise covariance matrix.

    See also
    --------
    For further details we refer to:

    - Conceptualization: M. Guardiani et al., Towards Moment-constrained
        Causal Modeling <https://doi.org/10.3390/psf2022005007> (Sec. 3.7).
    - Analytic geoVI parametrization: P. Frank et al., Geometric Variational
        Inference <https://doi.org/10.3390/e23070853> (Sec. 5.1)
    """
    if not isinstance(samples, Samples):
        raise TypeError("samples attribute should be of type `Samples`.")
    if not isinstance(likelihood, Likelihood):
        raise TypeError("likelhood is not an instance of `Likelihood`.")

    hamiltonian = StandardHamiltonian(likelihood)
    metric = hamiltonian.metric
    metric_size = jax.flatten_util.ravel_pytree(samples.pos)[0].size
    metric_linop, metric_matvec, metric_size = _ravel_metric(
        metric, samples.pos, dtype=likelihood.target.dtype, metric_jit=metric_jit
    )
    n_data_points = (
        size(likelihood.lsm_tangents_shape)
        if likelihood.lsm_tangents_shape is not None
        else metric_size
    )
    n_relevant_dofs = min(n_data_points, metric_size)

    trace_log_method = trace_log_method.lower()
    if trace_log_method not in ("eigsh", "slq"):
        raise ValueError("trace_log_method must be 'eigsh' or 'slq'.")
    trace_log_space = trace_log_space.lower()
    if trace_log_space not in ("auto", "signal", "data"):
        raise ValueError("trace_log_space must be 'auto', 'signal', or 'data'.")

    if trace_log_space == "data" and likelihood.lsm_tangents_shape is None:
        raise ValueError("trace_log_space='data' requires lsm_tangents_shape.")

    if trace_log_space == "data":
        use_data_space = True
    elif trace_log_space == "signal":
        use_data_space = False
    else:
        use_data_space = (
            likelihood.lsm_tangents_shape is not None and n_data_points <= metric_size
        )

    if use_data_space and verbose:
        logger.info(
            f"\nProjecting metric into data space for trace-log: "
            f"n_data={n_data_points}, n_signal={metric_size}."
        )

    space_suffix = "data" if use_data_space else "signal"
    save_eigensystem_prefix = f"{save_eigensystem_prefix}_{space_suffix}"

    if use_data_space:
        op_label = "data-space"
        op_linop, op_matvec, op_size = _make_data_operator(
            likelihood, samples.pos, dtype=likelihood.target.dtype, metric_jit=metric_jit
        )
        eigenvalue_shift = 0.0
        log_np = np.log1p
        log_f = jnp.log1p
    else:
        op_label = "metric"
        op_linop, op_matvec, op_size = metric_linop, metric_matvec, metric_size
        eigenvalue_shift = 1.0
        log_np = np.log
        log_f = jnp.log
    if compute_all:
        if verbose:
            logger.info(
                f"compute_all=True; computing all {n_relevant_dofs} relevant "
                f"eigenvalues."
            )
        n_eigenvalues = n_relevant_dofs

    eigenvalues, eigenvectors = _eigsh(
        op_linop,
        op_size,
        n_eigenvalues,
        tot_dofs=n_relevant_dofs,
        min_lh_eval=min_lh_eval,
        eigenvalue_shift=eigenvalue_shift,
        n_batches=n_batches,
        tol=tol,
        early_stop=not compute_all,
        verbose=verbose,
        output_directory=output_directory,
        save_eigensystem_prefix=save_eigensystem_prefix,
        resume_eigenvectors=resume_eigenvectors,
        resume_eigenvalues=resume_eigenvalues,
        orthonormalize_eigenvectors=orthonormalize_eigenvectors,
        orthonormalize_every_n_batches=orthonormalize_every_n_batches,
        orthonormalize_threshold=orthonormalize_threshold,
        orthonormalize_n_probes=orthonormalize_n_probes,
    )
    if eigenvalues is None:
        eigenvalues = np.asarray([], dtype=np.float64)
    if verbose:
        logger.info(
            f"\nComputed {eigenvalues.size} largest eigenvalues "
            f"in {op_label} space "
            f"(out of {n_relevant_dofs} relevant degrees of freedom)."
        )
        if not use_data_space and metric_size > n_relevant_dofs:
            logger.info(
                f"\nThe remaining {metric_size - n_relevant_dofs} metric "
                f"eigenvalues (out of {metric_size}) are equal to 1."
            )
        if eigenvalues.size > 0:
            logger.info(f"\n{eigenvalues}.")

    # Return a list of ELBO samples and a summary of the ELBO statistics
    log_eigenvalues = log_np(eigenvalues) if eigenvalues.size > 0 else np.array([])
    tr_log_lat_cov_lower = 0.0
    exact_log = np.sum(log_eigenvalues) if log_eigenvalues.size > 0 else 0.0
    slq_remainder = 0.0
    slq_remainder_se = 0.0
    slq_out = None

    tail_lo = None
    tail_hi = None

    trace_inv_exact = 0.0
    trace_inv_remainder = 0.0
    trace_inv_remainder_se = 0.0
    trace_inv_const = float(max(0, metric_size - n_relevant_dofs))
    prior_mean_sq = 0.0

    if analytic_prior_term:
        if trace_log_method == "eigsh" and n_relevant_dofs > log_eigenvalues.size:
            raise ValueError(
                "analytic_prior_term requires trace_log_method='slq' or "
                "compute_all=True when not all eigenvalues are computed."
            )
        if samples.pos is not None:
            mean = samples.pos
        elif len(samples) > 0:
            mean = tree_map(lambda x: jnp.mean(x, axis=0), samples.samples)
        else:
            mean = None
        if mean is not None:
            prior_mean_sq = float(np.asarray(vdot(mean, mean)))
        if eigenvalues.size > 0:
            if use_data_space:
                inv_eigs = 1.0 / (1.0 + eigenvalues)
            else:
                inv_eigs = 1.0 / eigenvalues
            trace_inv_exact = float(np.sum(inv_eigs))

    if trace_log_method == "eigsh":
        tr_log_lat_cov = -0.5 * np.sum(log_eigenvalues)
        if log_eigenvalues.size > 0:
            tr_log_lat_cov_lower = (
                0.5 * (n_relevant_dofs - log_eigenvalues.size)
                * np.min(log_eigenvalues)
            )
    else:
        if n_relevant_dofs > log_eigenvalues.size:
            if verbose:
                remaining = n_relevant_dofs - log_eigenvalues.size
                logger.info(
                    f"\nEstimating trace-log term for {remaining} eigenvalues "
                    f"via SLQ in {op_label} space."
                )
            if slq_key is None:
                slq_key = jax.random.PRNGKey(0)
            elif isinstance(slq_key, (int, np.integer)):
                slq_key = jax.random.PRNGKey(int(slq_key))
            else:
                slq_key = jnp.asarray(slq_key)
            if slq_order > op_size:
                if verbose:
                    logger.info(
                        f"slq_order={slq_order} exceeds operator size {op_size}; "
                        f"using slq_order={op_size}."
                    )
                slq_order = op_size
            slq_kwargs = {} if slq_kwargs is None else dict(slq_kwargs)
            forbidden = {
                "A",
                "f",
                "order",
                "num_samples",
                "key",
                "n",
                "deflate_eigvecs",
                "lam_min",
                "lam_max",
                "fixed_endpoint",
                "extra_fns",
            }
            overlap = forbidden.intersection(slq_kwargs)
            if overlap:
                raise ValueError(
                    f"slq_kwargs must not contain {sorted(overlap)}; "
                    "use the dedicated parameters instead."
                )
            lam_min = None
            lam_max = None
            if eigenvalues.size > 0:
                lam_min = float(eigenvalue_shift)
                lam_max = float(np.min(eigenvalues))
                if not np.isfinite(lam_max) or lam_max < lam_min:
                    lam_min = None
                    lam_max = None
            deflate = None if eigenvectors is None else jnp.asarray(eigenvectors)
            extra_fns = None
            if analytic_prior_term:
                if use_data_space:
                    def inv_f(x):
                        return 1.0 / (1.0 + x)
                else:
                    def inv_f(x):
                        return 1.0 / x
                extra_fns = {"inv": inv_f}

            if slq_jit:
                if deflate is None:
                    # JIT requires a concrete array; an empty basis means "no deflation".
                    deflate = jnp.zeros((op_size, 0), dtype=jnp.float64)
                if lam_min is None:
                    def _call_slq(key, deflate_eigvecs):
                        return slq_gauss_radau(
                            op_matvec,
                            log_f,
                            slq_order,
                            num_samples=slq_num_samples,
                            key=key,
                            n=op_size,
                            deflate_eigvecs=deflate_eigvecs,
                            extra_fns=extra_fns,
                            **slq_kwargs,
                        )
                else:
                    def _call_slq(key, deflate_eigvecs):
                        return slq_gauss_radau(
                            op_matvec,
                            log_f,
                            slq_order,
                            num_samples=slq_num_samples,
                            key=key,
                            n=op_size,
                            deflate_eigvecs=deflate_eigvecs,
                            lam_min=lam_min,
                            lam_max=lam_max,
                            extra_fns=extra_fns,
                            **slq_kwargs,
                        )
                slq_out = jax.jit(_call_slq)(slq_key, deflate)
            else:
                slq_out = slq_gauss_radau(
                    op_matvec,
                    log_f,
                    slq_order,
                    num_samples=slq_num_samples,
                    key=slq_key,
                    n=op_size,
                    deflate_eigvecs=deflate,
                    lam_min=lam_min,
                    lam_max=lam_max,
                    extra_fns=extra_fns,
                    **slq_kwargs,
                )
            slq_remainder = float(np.asarray(slq_out["estimate"]))
            slq_remainder_se = float(np.asarray(slq_out["stochastic_se"]))
            if analytic_prior_term and "extra_inv_estimate" in slq_out:
                trace_inv_remainder = float(np.asarray(slq_out["extra_inv_estimate"]))
                trace_inv_remainder_se = float(np.asarray(slq_out["extra_inv_se"]))
            if "radau_lo" in slq_out and "radau_hi" in slq_out:
                lo = float(np.asarray(slq_out["radau_lo"]))
                hi = float(np.asarray(slq_out["radau_hi"]))
                tail_lo = min(lo, hi)
                tail_hi = max(lo, hi)
        tr_log_lat_cov = -0.5 * (exact_log + slq_remainder)
        if use_radau_as_bound and tail_hi is not None:
            tr_log_lat_cov_lower = 0.5 * max(0.0, tail_hi - slq_remainder)
        else:
            tr_log_lat_cov_lower = 0.5 * slq_remainder_se
        if verbose and slq_out is not None:
            logger.info(
                f"\nTrace-log exact contribution: {exact_log:.4e}; "
                f"SLQ remainder: {slq_remainder:.4e} "
                f"(SE {slq_remainder_se:.2e})."
            )
    trace_inv_total = 0.0
    prior_term = 0.0
    if analytic_prior_term:
        trace_inv_total = trace_inv_exact + trace_inv_remainder + trace_inv_const
        prior_term = 0.5 * (trace_inv_total + prior_mean_sq)
        if trace_inv_remainder_se > 0.0:
            tr_log_lat_cov_lower += 0.5 * trace_inv_remainder_se
    posterior_contribution = tr_log_lat_cov + 0.5 * metric_size
    if analytic_prior_term:
        elbo_samples = np.array(
            list(posterior_contribution - likelihood(s) - prior_term for s in samples)
        )
    else:
        elbo_samples = np.array(
            list(posterior_contribution - hamiltonian(s) for s in samples)
        )

    stats = {"lower_error": tr_log_lat_cov_lower}
    if analytic_prior_term:
        stats.update(
            {
                "trace_inv_exact": float(trace_inv_exact),
                "trace_inv_slq": float(trace_inv_remainder),
                "trace_inv_se": float(trace_inv_remainder_se),
                "trace_inv_total": float(trace_inv_total),
                "prior_mean_sq": float(prior_mean_sq),
                "prior_term": float(prior_term),
            }
        )
    if trace_log_method == "slq":
        stats.update(
            {
                "trace_log_exact": float(exact_log),
                "trace_log_slq": float(slq_remainder),
                "trace_log_se": float(slq_remainder_se),
            }
        )
        if slq_out is not None:
            if "radau_estimate" in slq_out:
                stats["trace_log_radau"] = float(
                    np.asarray(slq_out["radau_estimate"])
                )
            if "radau_se" in slq_out:
                stats["trace_log_radau_se"] = float(np.asarray(slq_out["radau_se"]))
            if "quadrature_width" in slq_out:
                stats["trace_log_quadrature_width"] = float(
                    np.asarray(slq_out["quadrature_width"])
                )
        if tail_lo is not None and tail_hi is not None:
            stats["trace_log_tail_lo"] = float(tail_lo)
            stats["trace_log_tail_hi"] = float(tail_hi)
            stats["trace_log_quadrature_width"] = float(tail_hi - tail_lo)
            stats["trace_log_total_est"] = float(exact_log + slq_remainder)
            stats["trace_log_total_lo"] = float(exact_log + tail_lo)
            stats["trace_log_total_hi"] = float(exact_log + tail_hi)
    elbo_mean = np.mean(elbo_samples)
    elbo_std = np.std(elbo_samples, ddof=1)
    elbo_se = elbo_std / np.sqrt(len(samples)) if len(samples) > 0 else 0.0
    elbo_up = elbo_mean + elbo_std
    elbo_lw = elbo_mean - elbo_std - stats["lower_error"]
    stats["elbo_lw"], stats["elbo_mean"], stats["elbo_up"] = (
        elbo_lw,
        elbo_mean,
        elbo_up,
    )
    stats["elbo_std"] = elbo_std
    stats["elbo_se"] = elbo_se
    if verbose:
        if trace_log_method == "eigsh":
            remainder = n_relevant_dofs - log_eigenvalues.size
            tail = (
                remainder * np.min(log_eigenvalues)
                if log_eigenvalues.size > 0
                else 0.0
            )
            trace_msg = (
                f"\nTrace-log decomposition (in log units)"
                f"\nExact eigensum : {exact_log:.4e}"
                f"\nTail approx    : {tail:.4e} (n={remainder})"
            )
        else:
            trace_msg = (
                f"\nTrace-log decomposition (in log units)"
                f"\nExact eigensum : {exact_log:.4e}"
                f"\nSLQ remainder  : {slq_remainder:.4e} (SE {slq_remainder_se:.2e})"
            )
            if tail_lo is not None and tail_hi is not None:
                trace_msg += f"\nTail bounds    : [{tail_lo:.4e}, {tail_hi:.4e}]"
        logger.info(trace_msg)

        if analytic_prior_term:
            prior_msg = (
                f"\nAnalytic prior term (in log units)"
                f"\nTrace(inv) : {trace_inv_total:.4e} "
                f"(exact {trace_inv_exact:.4e}, "
                f"SLQ {trace_inv_remainder:.4e}, "
                f"const {trace_inv_const:.4e})"
                f"\nMean^2     : {prior_mean_sq:.4e}"
            )
            logger.info(prior_msg)

        s = (
            f"\nELBO decomposition (in log units)"
            f"\nELBO mean : {elbo_mean:.4e} (lower: {elbo_lw:.4e}, "
            f"upper: {elbo_up:.4e})"
            f"\nELBO std  : {elbo_std:.4e}"
        )
        logger.info(s)

    return elbo_samples, stats
