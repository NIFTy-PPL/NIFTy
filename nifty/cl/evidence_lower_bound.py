# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2022 Max-Planck-Society
# Author: Matteo Guardiani, Philipp Frank

import math
import os

import numpy as np
import scipy.linalg as slg
import scipy.sparse.linalg as ssl

from .field import Field
from .linearization import Linearization
from .logger import logger
from .minimization.sample_list import ResidualSampleList, SampleList
from .operator_spectrum import _DomRemover
from .operators.energy_operators import StandardHamiltonian
from .operators.sandwich_operator import SandwichOperator
from .sugar import makeField


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
    identity = np.identity(M.shape[0], dtype=np.float64)
    m = []
    for v in identity:
        m.append(M.matvec(v))
    return np.column_stack(m)


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


def _orthonormality_error(eigenvectors):
    if eigenvectors.size == 0:
        return 0.0
    gram = eigenvectors.conj().T @ eigenvectors
    gram -= np.eye(gram.shape[0], dtype=gram.dtype)
    return float(np.max(np.abs(gram)))


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


def _eigsh(
    metric,
    n_eigenvalues,
    tot_dofs,
    dtype,
    min_lh_eval=1e-4,
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
):
    metric = SandwichOperator.make(_DomRemover(metric.domain).adjoint, metric)
    metric_size = metric.domain.size
    M = ssl.LinearOperator(
        shape=2 * (metric_size,),
        matvec=lambda x: metric(makeField(metric.domain, x)).asnumpy(),
        dtype=dtype,
    )
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
        if resume_eigenvectors is not None and resume_eigenvalues is None:
            raise ValueError(
                "resume_eigenvalues is required when orthonormalize_eigenvectors=True."
            )

    if n_eigenvalues > tot_dofs:
        raise ValueError(
            "Number of requested eigenvalues exceeds the number of relevant degrees of freedom!"
        )

    batch_counter = 0
    if resume_eigenvectors is not None:
        eigenvectors = np.asarray(resume_eigenvectors)
        if eigenvectors.ndim != 2:
            raise ValueError("resume_eigenvectors must be a 2D array.")
        if eigenvectors.shape[0] != metric_size:
            raise ValueError("resume_eigenvectors does not match the metric size.")
        if resume_eigenvalues is None:
            eigenvalues = _estimate_eigenvalues(M, eigenvectors)
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
                _orthonormality_error(eigenvectors)
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
            "Number of provided eigenvectors exceeds relevant degrees of freedom!"
        )

    if (
        early_stop
        and eigenvalues is not None
        and abs(1.0 - np.min(eigenvalues)) < min_lh_eval
    ):
        return eigenvalues, eigenvectors

    n_precomputed = 0 if eigenvalues is None else eigenvalues.size
    remaining_eigenvalues = n_eigenvalues - n_precomputed
    if remaining_eigenvalues <= 0:
        return eigenvalues, eigenvectors

    if tot_dofs == n_eigenvalues and n_precomputed == 0:
        # Compute exact eigensystem
        if verbose:
            logger.info(f"Computing all {tot_dofs} relevant metric eigenvalues.")
        if output_directory is None:
            eigenvalues = slg.eigh(
                _explicify(M),
                eigvals_only=True,
                subset_by_index=[metric_size - tot_dofs, metric_size - 1],
            )
            eigenvalues = np.flip(eigenvalues)
        else:
            eigvals, eigvecs = slg.eigh(
                _explicify(M),
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
        # Set up batches
        base = remaining_eigenvalues // n_batches
        remainder = remaining_eigenvalues % n_batches
        batches = [base + 1] * remainder + [base] * (n_batches - remainder)
        batches = [batch for batch in batches if batch > 0]
        projected_metric = M
        if eigenvectors is not None:
            if orthonormalize_eigenvectors:
                error = (
                    _orthonormality_error(eigenvectors)
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
            projected_metric = projector @ M @ projector.T

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
            batch_counter += 1
            if orthonormalize_eigenvectors:
                error = (
                    _orthonormality_error(eigenvectors)
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
                        logger.info(
                            f"Re-orthonormalizing eigenvectors ({reason})."
                        )
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
                logger.info(
                    f"Eigenvalue progress: {done}/{n_eigenvalues} ({pct}%)"
                )

            if early_stop and abs(1.0 - np.min(eigenvalues)) < min_lh_eval:
                break
            # Project out subspace of already computed eigenvalues
            projector = _Projector(eigenvectors)
            projected_metric = projector @ M @ projector.T
    return eigenvalues, eigenvectors


def estimate_evidence_lower_bound(
    hamiltonian,
    samples,
    n_eigenvalues,
    *,
    compute_all=False,
    min_lh_eval=1e-3,
    n_batches=10,
    tol=0.0,
    verbose=True,
    dtype=np.float64,
    output_directory=None,
    save_eigensystem_prefix="metric",
    resume_eigenvectors=None,
    resume_eigenvalues=None,
    orthonormalize_eigenvectors=True,
    orthonormalize_every_n_batches=5,
    orthonormalize_threshold=1e-6,
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
    hamiltonian : :class:`nifty.cl.operators.energy_operators.StandardHamiltonian`
        Hamiltonian of the approximated probability distribution.
    samples : ResidualSampleList
        Collection of samples from the posterior distribution.
    n_eigenvalues : int
        Maximum number of eigenvalues to be considered for the estimation of
        the log-determinant of the metric. Note that if `n_eigenvalues` equals
        the total number of relevant degrees of freedom of the problem, all
        relevant eigenvalues are always computed irrespective of other stopping
        criteria.
    compute_all : bool
        If True, compute all eigenvalues and eigenvectors of the relevant
        metric subspace. Overrides `n_eigenvalues`.
    min_lh_eval : float
        Smallest eigenvalue of the likelihood to be considered. If the
        estimated eigenvalues become smaller then 1 + `min_lh_eval`, the
        eigenvalue estimation terminates and uses the smallest eigenvalue as a
        proxy for all remaining eigenvalues in the trace-log estimation.
        Default is 1e-3.
    n_batches : int
        Number of batches into which the eigenvalue estimation gets subdivided
        into. Only after completing one batch the early stopping criterion
        based on `min_lh_eval` is checked for.
    tol : Optional[float]
        Tolerance on the eigenvalue calculation. Zero indicates machine
        precision. Default is 0.
    verbose : Optional[bool]
        Print list of eigenvalues and summary of evidence calculation. Default
        is True.
    dtype : Optional[numpy.dtype]
        Data type of the eigenvalues and eigenvectors. Default is np.float64.
    output_directory : Optional[str]
        If set, saves the cumulative metric eigenvalues and eigenvectors after
        each batch to `{output_directory}/{prefix}_eigenvalues.npy` and
        `{output_directory}/{prefix}_eigenvectors.npy`.
    save_eigensystem_prefix : str
        Prefix for eigensystem filenames. Default is "metric".
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
        is True. Default is 5.
    orthonormalize_threshold : Optional[float]
        Re-orthonormalize whenever the maximum absolute deviation of
        `V.T @ V` from the identity exceeds this threshold. Set to `None` to
        disable this check. Default is 1e-6.

    Returns
    -------
    `elbo_samples` : SampleList
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
    quantities that are not explicitly dependent on the signal. Explicitly
    calculating these terms can be expensive, therefore they are usually
    neglected. Since in most cases they are also not required for model
    comparison, the provided estimate may not include terms which are constant
    in the signal. Only when comparing models with different noise statistics
    (or when the ELBO is needed to approximate the true evidence) these
    contributions have to be considered. For example, for a Gaussian
    distributed signal and a linear problem (Wiener Filter problem) the only
    term missing is :math:`-\\frac1 2 \\log \\det |2 \\pi N|`, where :math:`N`
    is the noise covariance matrix.

    See also
    --------
    For further details we refer to:

    * Analytic formulation: P. Frank et al., Geometric Variational Inference <https://arxiv.org/pdf/2105.10470.pdf> (Sec. 5.1)
    * Conceptualization: A. Kostić et al. (manuscript in preparation).
    """
    if not isinstance(samples, ResidualSampleList):
        raise TypeError("samples attribute should be of type ResidualSampleList.")
    if not isinstance(hamiltonian, StandardHamiltonian):
        raise TypeError("hamiltonian is not an instance of `ift.StandardHamiltonian`.")

    n_data_points = (
        hamiltonian.likelihood_energy.data_domain.size
        if not None
        else hamiltonian.domain.size
    )
    n_relevant_dofs = min(
        n_data_points, hamiltonian.domain.size
    )  # Number of metric eigenvalues that are not one

    metric = hamiltonian(Linearization.make_var(samples.mean, want_metric=True)).metric
    metric_size = metric.domain.size
    if compute_all:
        if verbose:
            logger.info(
                f"compute_all=True; computing all {n_relevant_dofs} relevant "
                f"eigenvalues."
            )
        n_eigenvalues = n_relevant_dofs

    eigenvalues, _ = _eigsh(
        metric,
        n_eigenvalues,
        n_relevant_dofs,
        dtype,
        min_lh_eval=min_lh_eval,
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
    )
    if verbose:
        # FIXME
        logger.info(
            f"\nComputed {eigenvalues.size} largest eigenvalues (out of {n_relevant_dofs} relevant degrees of freedom)."
            f"\nThe remaining {metric_size - n_relevant_dofs} metric eigenvalues (out of {metric_size}) are equal to "
            f"1.\n\n{eigenvalues}."
        )

    # Return a list of ELBO samples and a summary of the ELBO statistics
    log_eigenvalues = np.log(eigenvalues)
    tr_log_lat_cov = -0.5 * np.sum(log_eigenvalues)
    tr_log_lat_cov_lower = (
        0.5 * (n_relevant_dofs - log_eigenvalues.size) * np.min(log_eigenvalues)
    )
    tr_log_lat_cov_lower = Field.scalar(tr_log_lat_cov_lower)
    posterior_contribution = Field.scalar(tr_log_lat_cov + 0.5 * metric_size)
    elbo_samples = SampleList(
        list(samples.iterator(lambda x: posterior_contribution - hamiltonian(x)))
    )

    stats = {"lower_error": tr_log_lat_cov_lower}
    elbo_mean, elbo_var = elbo_samples.sample_stat()
    elbo_up = elbo_mean + elbo_var.sqrt()
    elbo_lw = elbo_mean - elbo_var.sqrt() - stats["lower_error"]
    stats["elbo_mean"], stats["elbo_up"], stats["elbo_lw"] = elbo_mean, elbo_up, elbo_lw
    if verbose:
        s = (
            f"\nELBO decomposition (in log units)"
            f"\nELBO mean : {elbo_mean.asnumpy():.4e} "
            f"(lower: {elbo_lw.asnumpy():.4e}, "
            f"upper: {elbo_up.asnumpy():.4e})"
        )
        logger.info(s)

    return elbo_samples, stats
