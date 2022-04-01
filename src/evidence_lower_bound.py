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
import numpy as np
import scipy.linalg as slg
import scipy.sparse.linalg as ssl

from .linearization import Linearization
from .logger import logger
from .minimization.sample_list import ResidualSampleList, SampleList
from .operator_spectrum import _DomRemover
from .operators.sandwich_operator import SandwichOperator
from .sugar import makeField
from .field import Field


class _Projector(ssl.LinearOperator):
    """
    Computes the projector of a Matrix or LinearOperator as a LinearOperator given the eigenvectors of the
    complementary space.

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
        super().__init__(np.dtype('f8'), 2 * (eigenvectors.shape[0],))
        self.eigenvectors = eigenvectors

    def _matvec(self, x):
        res = x.copy()
        for eigenvector in self.eigenvectors.T:
            res -= eigenvector * eigenvector.dot(x)
        return res

    def _rmatvec(self, x):
        return self._matvec(x)

def _explicify(M):
    identity = np.identity(M.shape[0], dtype=np.float64)
    m = []
    for v in identity:
        m.append(M.matvec(v))
    return np.vstack(m).T

def _eigsh(metric, n_eigenvalues, min_lh_eval = 1E-3, batch_number = 10,
           tol = 0., verbose = True):
    metric = SandwichOperator.make(_DomRemover(metric.domain).adjoint, metric)
    M = ssl.LinearOperator(
            shape = 2 * (metric.domain.size,),
            matvec = lambda x: metric(makeField(metric.domain, x)).val)

    if n_eigenvalues > metric.domain.size:
        raise ValueError("Number of requested eigenvalues exeeds size of matrix!")

    if metric.domain.size == n_eigenvalues:
        # Compute exact eigensystem 
        if verbose:
            logger.info(f"Number of eigenvalues being computed: {n_eigenvalues}")
        eigenvalues, eigenvectors = slg.eigh(_explicify(M))
    else:
        # Set up batches
        batchsize = int(n_eigenvalues / batch_number)
        batches = [batchsize, ]*(batch_number-1) 
        batches += [n_eigenvalues - batchsize*(batch_number - 1), ]
        eigenvalues, eigenvectors, Mproj = None, None, M
        for batch in batches:
            if verbose:
                logger.info(f"Number of eigenvalues being computed: {batch}")
            # Get eigensystem for current batch
            eigvals, eigvecs = ssl.eigsh(Mproj, k=batch, tol=tol,
                                        return_eigenvectors=True, which='LM')
            eigenvalues = eigvals if eigenvalues is None else np.concatenate((eigenvalues, eigvals))
            eigenvectors = eigvecs if eigenvectors is None else np.hstack((eigenvectors, eigvecs))

            if abs(1.0 - np.min(eigenvalues)) < min_lh_eval:
                break
            # Project out subspace of already computed eigenvalues
            projector = _Projector(eigenvectors)
            Mproj = projector @ M @ projector.T
    return eigenvalues, eigenvectors

def estimate_evidence_lower_bound(hamiltonian, samples, n_eigenvalues,
                                  min_lh_eval = 1E-3, batch_number = 10,
                                  tol = 0., verbose = True):
    """Provides an estimate for the Evidence Lower Bound (ELBO).

    Statistical inference deals with the problem of hypothesis testing, given the data and models that can describe it.
    In general, it is hard to find a good metric to discern between different models. In Bayesian Inference,
    the Bayes factor can serve this purpose.
    To compute the Bayes factor it is necessary to calculate the evidence, given the specific model :math:`p(
    d|\\text{model})` at hand.
    Then, the ratio between the evidence of a model A and the one of a model B represents how much more likely it is
    that model A represents the data better than model B.

    The evidence for an approximated-inference problem could be in principle calculated by considering

    .. math ::
        \\log(p(d)) - D_\\text{KL} \\left[ Q(\\theta(\\xi)|d) || p(\\theta(\\xi) | d) \\right] = -\\langle H(\\theta(
        \\xi), d)\\rangle + \\frac1 2 \\left( N + \\text{Tr } \\log\\Lambda\\right)

    where :math:`D_\\text{KL} \\left[ Q || p \\right]` is the Kullback-Leibler (KL) divergence between the
    approximating posterior distribution :math:`Q` and the actual posterior :math:`p`.
    But since the Kullback-Leibler divergence :math:`D_\\text{KL} [\\cdot, \\cdot] \\geq 0` is positive definite,
    it is convenient to consider the lower bound

    .. math ::
        \\log(p(d)) \\geq -\\langle H(\\theta(\\xi), d)\\rangle + \\frac1 2 \\left(N + \\text{Tr } \\log\\Lambda
        \\right),

    which takes the name of Evidence Lower Bound (ELBO).

    If the KL divergence is well minimized (which should always be the case when a Variational Inference approach is
    followed), then it is possible to utilize the ELBO (as a proxy for the actual evidences) and calculate the Bayes
    factors for model comparison.


    Parameters
    ----------

    hamiltonian : StandardHamiltonian
        Hamiltonian of the approximated probability distribution.

    samples : ResidualSampleList
        Collection of samples from the posterior distribution.

    n_eigenvalues : int
        Maximum number of eigenvalues to be considered for the estimation of the
        log determinant of the metric. Note that if `n_eigenvalues` equals the
        total number of dimensions of the problem, all eigenvalues are always 
        computed irrespective of other stopping criteria.

    min_lh_eval : float
        Smallest eigenvalue of the likelihood to be considered. If the estimated
        eigenvalues become smaller then 1 + `min_lh_eval`, the eigenvalue
        estimation terminates and uses the smallest eigenvalue as a proxy for
        all remaining eigenvalues in the trace log estimation. Default is 1e-3.

    batch_number : int
        Number of batches into which the eigenvalue estimation gets subdivided
        into. Only after completing one batch the early stopping criterion based
        on `min_lh_eval` is checked for.

    tol : Optional[float]
        Tolerance on the eigenvalue calculation. Zero indicates machine precision.
        Default is 0.

    verbose : Optional[bool]
        FIXME
        Annoy me with what you are doing. Or don't.
        Default is True.

    Returns
    ----------
    stats : dict
        Dictionary with the statistics of the estimated ELBO.


    Notes
    -----
    FIXME
    **IMPORTANT**: The provided estimate is missing the constant term :math:`-\\frac1 2 \\log \\det |2 \\pi N|`,
    where :math:`N` is the noise covariance matrix. This term is not considered in (most of) the NIFTy
    implementations of the energy operators, since it is constant throughout minimization. To obtain the actual ELBO
    this term should be added. Since calculating this term can in principle be expensive and in most cases it is
    not needed for model comparison (the noise statistics should be independent of the model at hand) it has here
    been neglected.
    Only when comparing models with different noise statistics (or when the ELBO is needed to approximate the true
    evidence) this contribution should be added.

    It is advisable to start with a higher number of initial eigenvalues to ensure better convergence.

    See also
    -----
    For further details on the analytic formulation we refer to A. Kostić et Al.
    (manuscript in preparation).
    """

    # TODO: Implement checks on the input parameters
    if not isinstance(samples, ResidualSampleList):
        raise TypeError("samples attribute should be of type ResidualSampleList.")

    metric = hamiltonian(Linearization.make_var(samples._m, want_metric=True)).metric
    metric_size = metric.domain.size
    eigenvalues, _ = _eigsh(metric, n_eigenvalues, min_lh_eval = min_lh_eval,
                            batch_number = batch_number, tol = tol, verbose = verbose)
    if verbose:
        # FIXME
        logger.info(f"{eigenvalues.size} largest eigenvalues (out of {metric_size})\n{eigenvalues}")

    # Calculate the \Tr \log term upper bound
    log_eigenvalues = np.log(eigenvalues)
    tr_log_lat_cov = - 0.5 * np.sum(log_eigenvalues) 
    # And its lower bound
    tr_log_lat_cov_lower = 0.5 * (metric_size - log_eigenvalues.size) * np.min(log_eigenvalues)
    tr_log_lat_cov_lower = Field.scalar(tr_log_lat_cov_lower)

    elbo = Field.scalar(tr_log_lat_cov + 0.5 * metric_size)
    elbo_samples = SampleList(list([-h + elbo for h in samples.iterator(hamiltonian)]))
    
    #FIXME
    stats = {'lower_error': tr_log_lat_cov_lower}
    if verbose:
        #FIXME
        s = (f"\nELBO decomposition (in log units)"
            )
        logger.info(s)

    return elbo_samples, stats
