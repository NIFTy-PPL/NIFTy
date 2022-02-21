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
import scipy.sparse.linalg as ssl

from .field import Field
from .logger import logger
from .minimization.kl_energies import SampledKLEnergyClass
from .minimization.sample_list import SampleListBase
from .operator_spectrum import _DomRemover
from .operators.linear_operator import LinearOperator
from .operators.sandwich_operator import SandwichOperator
from .sugar import makeField


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


def _return_eigenspace(M, n_eigenvalues, tol):
    eigenvalues, eigenvectors = ssl.eigsh(M, k=n_eigenvalues, tol=tol, return_eigenvectors=True, which='LM')
    i = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = np.flip(eigenvalues[i]), np.flip(eigenvectors[:, i], axis=1)
    return eigenvalues, eigenvectors


def estimate_evidence_lower_bound(mean, samples, n_eigenvalues, hamiltonian, constants=[], invariants=None,
                                  fudge_factor=0,
                                  max_iteration=5, eps=1e-3, tol=0., verbose=True, data=None):
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
        \\log(p(d)) - D_\\text{KL} \\left[ Q(\\theta(\\xi)|d) || p(\\theta(\\xi) | d) \\right] = -\\langleH(
        d|\\xi)\\rangle - \\frac{1}{2} \\left( \\Tr[\\Lambda - \\mathbb{1}] + \\xi^\\dagger \\xi - \\Tr\\log\\Lambda
        \\right)

    where :math:`D_\\text{KL} \\left[ Q || p \\right]` is the Kullback-Leibler (KL) divergence between the
    approximating posterior distribution :math:`Q` and the actual posterior :math:`p`.
    But since the Kullback-Leibler divergence :math:`D_\\text{KL}[\\dot] \\geq 0` is positive definite, it is convenient
    to consider the lower bound

    .. math ::
        \\log(p(d)) \\geq -\\langleH(d|\\xi)\\rangle - \\frac{1}{2} \\left( \\Tr[\\Lambda - \\mathbb{1}] +
        \\xi^\\dagger \\xi - \\Tr\\log\\Lambda \\right),

    which takes the name of Evidence Lower Bound (ELBO).

    If the KL divergence is well minimized (which should always be the case when a Variational Inference approach is
    followed), then it is possible to utilize the ELBO (as a proxy for the actual evidences) and calculate the Bayes
    factors for model comparison.


    Parameters
    ----------

    mean : Field
        Position at which the metric is to be evaluated.

    samples : SampleListBase
        Collection of samples from the posterior distribution.

    hamiltonian : StandardHamiltonian
        Hamiltonian of the approximated probability distribution.

    constants : list
        Collection of parameter keys that are kept constant during optimization.
        Default is no constants.

    invariants : XXX
        TODO: Explain
        Default is None.

    n_eigenvalues : int
        The starting number of eigenvalues and eigenvectors to be calculated.
        `n_eigenvalues` must be smaller than N-1, where N is the metric dimension.

    fudge_factor : signed int
        Constant number of eigenvalues that will always be subtracted (or
        computed) during each iteration of the algorithm.

    max_iteration : int
        Maximum iteration -> stopping criterion for the algorithm.

    eps : float
        Relative accuracy (with respect to `1.`) for the eigenvalues (stopping
        criterion).

    tol : float
        Tolerance on the eigenvalue calculation. Zero indicates machine precision.

    verbose : bool
        Annoy me with what you are doing. Or don't.
        Default is True.

    data :
        Required to evaluate evidence contributions deriving from
        parameter-independent likelihood terms that might have been ignored
        during minimization.


    Returns
    ----------
    stats : dict
        Dictionary with the statistics of the estimated ELBO.


    Notes
    -----
    It is advisable to start with a higher number of initial eigenvalues to ensure better convergence.

    See also
    -----
    For further details on the analytic formulation we refer to A. Kostić et Al.
    (manuscript in preparation).
    """

    # TODO: Implement checks on the input parameters
    if not isinstance(samples, SampleListBase):
        raise TypeError("samples attribute should be of type SampleListBase.")

    if verbose:
        logger.info(f"Number of eigenvalues to compute: {n_eigenvalues}")

    metric = SampledKLEnergyClass(samples, hamiltonian, constants, invariants,
                                  False).metric  # FIXME: Check parameters of SKLEC

    Ar = SandwichOperator.make(_DomRemover(metric.domain).adjoint, metric)
    M = ssl.LinearOperator(shape=2 * (metric.domain.size,), matvec=lambda x: Ar(makeField(Ar.domain, x)).val)

    eigenvalues, eigenvectors = _return_eigenspace(M, n_eigenvalues, tol)

    eigenvalue_error = abs(1.0 - eigenvalues[-1])
    count = 0

    while (eigenvalue_error > eps) and (count < max_iteration):
        projected_metric = _Projector(eigenvectors) @ M if count == 0 else _Projector(eigenvectors) @ projected_metric
        n_eigenvalues -= n_eigenvalues // 4 + fudge_factor  # FIXME: Not sure if it makes sense. Open to suggestions
        if verbose:
            logger.info(f"Number of additional eigenvalues being computed: {n_eigenvalues}")

        eigvals, eigenvectors = _return_eigenspace(projected_metric, n_eigenvalues, tol)
        eigenvalues = np.concatenate((eigenvalues, eigvals))
        eigenvalue_error = abs(1.0 - eigenvalues[-1])

        if eigenvalue_error == 0.:
            break
        count += 1

    if verbose:
        logger.info(f"{eigenvalues.size} largest eigenvalues\n{eigenvalues}")

    # Calculate the \Tr \ln term and \Tr terms
    tr_reduced_diagonal_lat_cov = 0.
    tr_log_diagonal_lat_cov = 0.
    for ev in eigenvalues:
        # The eigenvalues are of Theta^{-1} and we need the 1/ev for the ELBO calculation
        ev_lat_cov = 1. / ev
        # Error assessment made here is very conservative
        if abs(ev_lat_cov - 1.) > eps:
            tr_reduced_diagonal_lat_cov += ev_lat_cov - 1.
        if abs(np.log(ev_lat_cov)) > eps:
            tr_log_diagonal_lat_cov += np.log(ev_lat_cov)

    # Asses maximal error: propagate the error on the eigenvalues
    unexplored_dimensions = metric.domain.size - n_eigenvalues
    tr_reduced_diagonal_lat_cov_error = unexplored_dimensions * (min(eigenvalues) - 1.)
    tr_log_diagonal_lat_cov_error = unexplored_dimensions * np.log(min(eigenvalues))

    # Calculate the contribution from the prior
    squared_mean = mean.vdot(mean).val
    prior_evidence = 0.5 * (squared_mean + tr_reduced_diagonal_lat_cov)

    hamiltonian_mean, hamiltonian_var = samples.sample_stat(hamiltonian)
    hamiltonian_mean_std = np.sqrt(hamiltonian_var.val / samples.n_samples)  # std of the estimate for the mean

    h0 = 0.
    # if isinstance(hamiltonian.likelihood_energy, ift.PoissonianEnergy): # FIXME: Try to simplify this expression
    likelihood_type = str(hamiltonian.likelihood_energy.__dict__['_ops'][0])

    if likelihood_type in {'PoissonianEnergy', 'PoissonianEnergy ()'}:
        from scipy.special import factorial
        data = data.to_numpy() if hasattr(data, "to_numpy") else data.val if hasattr(data, "val") else data
        h0 = np.log(factorial(data)).sum()

    # elif not isinstance(hamiltonian.likelihood_energy, ift.GaussianEnergy): # FIXME: Try to simplify this expression
    elif not likelihood_type in {'GaussianEnergy', 'GaussianEnergy ()'}:
        tp_lh = type(hamiltonian.likelihood_energy)
        warn_msg = (f"Unknown likelihood of type {tp_lh!r};\n"
                    f"The (log) ELBO is potentially missing some constant term from the likelihood.")
        logger.warn(warn_msg)

    # TODO: Implement these checks for all NIFTy-supported likelihoods

    elbo_mean = - h0 - hamiltonian_mean - prior_evidence + 0.5 * tr_log_diagonal_lat_cov
    elbo_var_upper = abs(
        hamiltonian_mean_std - 0.5 * tr_reduced_diagonal_lat_cov_error + 0.5 * tr_log_diagonal_lat_cov_error)
    elbo_var_lower = abs(
        - hamiltonian_mean_std - 0.5 * tr_reduced_diagonal_lat_cov_error + 0.5 * tr_log_diagonal_lat_cov_error)

    stats = {"estimate": elbo_mean, "upper": elbo_mean + elbo_var_upper, "lower": elbo_mean - elbo_var_lower,
             "ln_likelihood_mean": hamiltonian_mean, "ln_likelihood_mean_std": hamiltonian_mean_std,
             "ln_likelihood_0": h0, "xi^2": squared_mean, "tr_reduced_diag_lat_cov": tr_reduced_diagonal_lat_cov,
             "tr_reduced_diagonal_lat_cov_error": tr_reduced_diagonal_lat_cov_error,
             "tr_ln_diag_lat_cov": tr_log_diagonal_lat_cov,
             "tr_log_diagonal_lat_cov_error": tr_log_diagonal_lat_cov_error}

    if verbose:
        s = (f"\nELBO decomposition (in log units)"
             f"\nELBO           : {stats['estimate'].val:.4e} (upper: {stats['upper'].val:.4e}, lower: "
             f"{stats['lower'].val:.4e})"
             f"\nH_lh           : {stats['ln_likelihood_mean'].val:.4e} ± {stats['ln_likelihood_mean_std']:.5e}"
             f"\nH_{{0, lh}}       : {stats['ln_likelihood_0']:.4e} "
             f"\n\\xi^2         : {stats['xi^2']:.4e}"
             f"\nTr \\Lambda     : {stats['tr_reduced_diag_lat_cov']:.5e} (+ "
             f"{stats['tr_reduced_diagonal_lat_cov_error']:.5e})"
             f"\nTr \\log \\Lambda : {stats['tr_ln_diag_lat_cov']:.5e} (+ {stats['tr_log_diagonal_lat_cov_error']:.5e})")
        logger.info(s)
    return stats
