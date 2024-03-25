#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax.flatten_util
import numpy as np
import scipy.linalg as slg
import scipy.sparse.linalg as ssl

from .evi import Samples
from .likelihood import Likelihood
from .logger import logger
from .optimize_kl import _StandardHamiltonian as StandardHamiltonian
from .tree_math.vector_math import size


class _Projector(ssl.LinearOperator):
    """Computes the projector of a Matrix or LinearOperator as a LinearOperator
    given the eigenvectors of the complementary space.

    Parameters
    ----------
    eigenvectors : ndarray
        The eigenvectors representing the directions to project out.

    Returns
    -------
    Projector : LinearOperator
        Operator representing the projection.
    """

    def __init__(self, eigenvectors):
        super().__init__(eigenvectors.dtype, 2 * (eigenvectors.shape[0],))
        self.eigenvectors = eigenvectors

    def _matvec(self, x):
        res = x.copy()
        for eigenvector in self.eigenvectors.T:
            res -= eigenvector * np.vdot(eigenvector, x)
        return res

    def _rmatvec(self, x):
        return self._matvec(x)


def _explicify(M):
    n = M.shape[0]
    m = []
    for i in range(n):
        basis_vector = np.zeros(n)
        basis_vector[i] = 1
        m.append(M @ basis_vector)
    return np.stack(m, axis=1)


def _ravel_metric(metric, position, dtype):
    shape = 2 * (size(metric(position, position)),)

    ravel = lambda x: jax.flatten_util.ravel_pytree(x)[0]
    unravel = lambda x: jax.linear_transpose(ravel, position)(x)[0]
    met = lambda x: ravel(metric(position, unravel(x)))

    return ssl.LinearOperator(shape=shape, dtype=dtype, matvec=met)


def _eigsh(
    metric,
    n_eigenvalues,
    tot_dofs,
    min_lh_eval=1e-4,
    batch_size=10,
    tol=0.0,
    verbose=True,
):
    metric_size = metric.shape[0]
    eigenvectors = None
    if n_eigenvalues > tot_dofs:
        raise ValueError(
            "Number of requested eigenvalues "
            "exceeds the number of relevant degrees of freedom!"
        )

    if tot_dofs == n_eigenvalues:
        # Compute exact eigensystem
        if verbose:
            logger.info(f"Computing all {tot_dofs} relevant metric eigenvalues.")
        eigenvalues = slg.eigh(
            _explicify(metric),
            eigvals_only=True,
            subset_by_index=[metric_size - tot_dofs, metric_size - 1],
        )
        eigenvalues = np.flip(eigenvalues)
    else:
        # Set up batches
        batch_size = n_eigenvalues // batch_size
        batches = [
            batch_size,
        ] * (batch_size - 1)
        batches += [
            n_eigenvalues - batch_size * (batch_size - 1),
        ]
        eigenvalues, projected_metric = None, metric
        for batch in batches:
            if verbose:
                logger.info(f"\nNumber of eigenvalues being computed: {batch}")
            # Get eigensystem for current batch
            eigvals, eigvecs = ssl.eigsh(
                projected_metric, k=batch, tol=tol, return_eigenvectors=True, which="LM"
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

            if abs(1.0 - np.min(eigenvalues)) < min_lh_eval:
                break
            # Project out subspace of already computed eigenvalues
            projector = _Projector(eigenvectors)
            projected_metric = projector @ metric @ projector.T
    return eigenvalues, eigenvectors


def estimate_evidence_lower_bound(
    likelihood,
    samples,
    n_eigenvalues,
    min_lh_eval=1e-3,
    batch_size=10,
    tol=0.0,
    verbose=True,
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
    likelihood : :class:`nifty8.re.likelihood.Likelihood`
        Log-likelihood of the model.
    samples : :class:`nifty8.re.evi.Samples`
        Collection of samples from the posterior distribution.
    n_eigenvalues : int
        Maximum number of eigenvalues to be considered for the estimation of
        the log-determinant of the metric. Note that if `n_eigenvalues` equals
        the total number of relevant degrees of freedom of the problem, all
        relevant eigenvalues are always computed irrespective of other stopping
        criteria.
    min_lh_eval : float
        Smallest eigenvalue of the likelihood to be considered. If the
        estimated eigenvalues become smaller than 1 + `min_lh_eval`, the
        eigenvalue estimation terminates and uses the smallest eigenvalue as a
        proxy for all remaining eigenvalues in the trace-log estimation.
        Default is 1e-3.
    batch_size : int
        Number of batches into which the eigenvalue estimation gets subdivided
        into. Only after completing one batch the early stopping criterion
        based on `min_lh_eval` is checked for.
    tol : Optional[float]
        Tolerance on the eigenvalue calculation. Zero indicates machine
        precision. Default is 0.
    verbose : Optional[bool]
        Print list of eigenvalues and summary of evidence calculation. Default
        is True.

    Returns
    -------
    `elbo_samples` : np.array
        List of elbo samples from the posterior distribution. The samples are
        returned to allow for more accurate elbo statistics.
    stats : dict
        Dictionary with a summary of the statistics of the estimated ELBO.
        The keys of this dictionary are:

        - `elbo_mean`: returns the mean value of the elbo estimate calculated
          over posterior samples
        - `elbo_up`: returns an upper bound to the elbo estimate (given by one
          posterior-sample standard deviation)
        - `elbo_lw`: returns a lower bound to the elbo estimate (one standard
          deviation plus a maximal error on the metric trace-log)
        - `lower_error`: maximal error on the metric trace-log term given by
          the number of relevant metric eigenvalues different from 1 neglected
          in the estimation of the trace-log times the log of the smallest
          calculated eigenvalue.

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

    * Analytic geoVI parametrization: P. Frank et al., Geometric Variational Inference
    <https://arxiv.org/pdf/2105.10470.pdf> (Sec. 5.1)
    * Conceptualization: A. Kostić et al. (manuscript in preparation).
    """
    if not isinstance(samples, Samples):
        raise TypeError("samples attribute should be of type `Samples`.")
    if not isinstance(likelihood, Likelihood):
        raise TypeError("likelhood is not an instance of `Likelihood`.")

    hamiltonian = StandardHamiltonian(likelihood)
    metric = hamiltonian.metric
    metric = _ravel_metric(metric, samples.pos, dtype=likelihood.target.dtype)
    metric_size = metric.shape[0]
    n_data_points = likelihood.lsm_tangents_shape.size if not None else metric_size
    n_relevant_dofs = min(
        n_data_points, metric_size
    )  # Number of metric eigenvalues that are not 1

    eigenvalues, _ = _eigsh(
        metric,
        n_eigenvalues,
        tot_dofs=n_relevant_dofs,
        min_lh_eval=min_lh_eval,
        batch_size=batch_size,
        tol=tol,
        verbose=verbose,
    )
    if verbose:
        logger.info(
            f"\nComputed {eigenvalues.size} largest eigenvalues (out of {n_relevant_dofs} "
            f"relevant degrees of freedom)."
            f"\nThe remaining {metric_size - n_relevant_dofs} metric eigenvalues "
            f"(out of {metric_size}) are equal to "
            f"1.\n\n{eigenvalues}."
        )

    # Return a list of ELBO samples and a summary of the ELBO statistics
    log_eigenvalues = np.log(eigenvalues)
    tr_log_lat_cov = -0.5 * np.sum(log_eigenvalues)
    tr_log_lat_cov_lower = (
        0.5 * (n_relevant_dofs - log_eigenvalues.size) * np.min(log_eigenvalues)
    )
    posterior_contribution = tr_log_lat_cov + 0.5 * metric_size
    elbo_samples = np.array(
        list(posterior_contribution - hamiltonian(s) for s in samples)
    )

    stats = {"lower_error": tr_log_lat_cov_lower}
    elbo_mean = np.mean(elbo_samples)
    elbo_std = np.std(elbo_samples, ddof=1)
    elbo_up = elbo_mean + elbo_std
    elbo_lw = elbo_mean - elbo_std - stats["lower_error"]
    stats["elbo_mean"], stats["elbo_up"], stats["elbo_lw"] = elbo_mean, elbo_up, elbo_lw
    if verbose:
        s = (
            f"\nELBO decomposition (in log units)"
            f"\nELBO mean : {elbo_mean:.4e} (upper: {elbo_up:.4e}, lower: {elbo_lw:.4e})"
        )
        logger.info(s)

    return elbo_samples, stats
