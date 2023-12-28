import jax.flatten_util
import numpy as np
import scipy.linalg as slg
import scipy.sparse.linalg as ssl

import nifty8.re as jft
from nifty8.re.logger import logger


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


def ravel_metric(metric, position):
    dim = 0
    shapes = jft.tree_shape(metric(position, position).tree)
    for s in shapes.values():
        dim += np.prod(s)
    shape = 2*(int(dim),)

    ravel = lambda x: jax.flatten_util.ravel_pytree(x)[0]
    unravel = lambda x: jax.linear_transpose(ravel, position)(x)[0]
    met = lambda x: ravel(metric(position, unravel(x)))

    return ssl.LinearOperator(shape=shape, matvec=met)


def _eigsh(metric, n_eigenvalues, tot_dofs, min_lh_eval=1e-4, batch_number=10, tol=0., verbose=True):
    metric_size = metric.shape[0]
    eigenvectors = None
    if n_eigenvalues > tot_dofs:
        raise ValueError("Number of requested eigenvalues "
                         "exceeds the number of relevant degrees of freedom!")

    if tot_dofs == n_eigenvalues:
        # Compute exact eigensystem
        if verbose:
            logger.info(f"Computing all {tot_dofs} relevant metric eigenvalues.")
        eigenvalues = slg.eigh(_explicify(metric), eigvals_only=True,
                               subset_by_index=[metric_size - tot_dofs, metric_size - 1])
        eigenvalues = np.flip(eigenvalues)
    else:
        # Set up batches
        batch_size = n_eigenvalues // batch_number
        batches = [batch_size, ] * (batch_number - 1)
        batches += [n_eigenvalues - batch_size * (batch_number - 1), ]
        eigenvalues, projected_metric = None, metric
        for batch in batches:
            if verbose:
                logger.info(f"\nNumber of eigenvalues being computed: {batch}")
            # Get eigensystem for current batch
            eigvals, eigvecs = ssl.eigsh(projected_metric, k=batch, tol=tol,
                                         return_eigenvectors=True, which='LM')
            i = np.argsort(eigvals)
            eigvals, eigvecs = np.flip(eigvals[i]), np.flip(eigvecs[:, i], axis=1)
            eigenvalues = eigvals if eigenvalues is None else np.concatenate((eigenvalues, eigvals))
            eigenvectors = eigvecs if eigenvectors is None else np.hstack((eigenvectors, eigvecs))

            if abs(1.0 - np.min(eigenvalues)) < min_lh_eval:
                break
            # Project out subspace of already computed eigenvalues
            projector = _Projector(eigenvectors)
            projected_metric = projector @ metric @ projector.T
    return eigenvalues, eigenvectors


def estimate_evidence_lower_bound(hamiltonian, samples, n_eigenvalues, min_lh_eval=1e-3,
                                  batch_number=10, tol=0., verbose=True):
    # if not isinstance(samples, ResidualSampleList):
    #     raise TypeError("samples attribute should be of type ResidualSampleList.")
    # if not isinstance(hamiltonian, StandardHamiltonian):
    #     raise TypeError("hamiltonian is not an instance of `ift.StandardHamiltonian`.")

    metric = hamiltonian.metric
    metric = ravel_metric(metric, samples.pos) #FIXME pos is correct?
    metric_size = metric.shape[0]
    n_data_points = hamiltonian.likelihood.domain.size if not None else metric_size
    n_relevant_dofs = min(n_data_points, metric_size)  # Number of metric eigenvalues that are not one


    eigenvalues, _ = _eigsh(metric, n_eigenvalues, tot_dofs=n_relevant_dofs, min_lh_eval=min_lh_eval,
                            batch_number=batch_number, tol=tol, verbose=verbose)
    if verbose:
        # FIXME
        logger.info(
            f"\nComputed {eigenvalues.size} largest eigenvalues (out of {n_relevant_dofs} relevant degrees of freedom)."
            f"\nThe remaining {metric_size - n_relevant_dofs} metric eigenvalues (out of {metric_size}) are equal to "
            f"1.\n\n{eigenvalues}.")

    # Return a list of ELBO samples and a summary of the ELBO statistics
    log_eigenvalues = np.log(eigenvalues)
    tr_log_lat_cov = - 0.5 * np.sum(log_eigenvalues)
    tr_log_lat_cov_lower = 0.5 * (n_relevant_dofs - log_eigenvalues.size) * np.min(log_eigenvalues)
    posterior_contribution = tr_log_lat_cov + 0.5 * metric_size
    elbo_samples = list(posterior_contribution - hamiltonian(s) for s in samples)

    stats = {'lower_error': tr_log_lat_cov_lower}
    elbo_mean = np.mean(elbo_samples)
    elbo_var = np.std(elbo_samples, ddof=1)
    elbo_up = elbo_mean + np.sqrt(elbo_var)
    elbo_lw = elbo_mean - np.sqrt(elbo_var) - stats["lower_error"]
    stats['elbo_mean'], stats['elbo_up'], stats['elbo_lw'] = elbo_mean, elbo_up, elbo_lw
    if verbose:
        s = (f"\nELBO decomposition (in log units)"
             f"\nELBO mean : {elbo_mean:.4e} (upper: {elbo_up:.4e}, lower: {elbo_lw:.4e})")
        logger.info(s)

    return elbo_samples, stats