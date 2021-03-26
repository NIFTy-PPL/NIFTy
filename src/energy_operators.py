from jax import numpy as np
from jax.tree_util import tree_map

from .operator import Likelihood, ShapeWithDtype
from .optimize import cg
from .sugar import random_like, sum_of_squares


class StandardHamiltonian(Likelihood):
    def __init__(self, likelihood, _compile_joined=False):
        self._nll = likelihood

        def joined_hamiltonian(primals):
            return self._nll(primals) + 0.5 * sum_of_squares(primals)

        def joined_metric(primals, tangents):
            return self._nll.metric(primals, tangents) + tangents

        if _compile_joined:
            from jax import jit
            joined_hamiltonian = jit(joined_hamiltonian)
            joined_metric = jit(joined_metric)
        self._hamiltonian = joined_hamiltonian
        self._metric = joined_metric
        # We do not know the shape of the tangent space of the left_sqrt_metric
        self._left_sqrt_metric = None
        self._lsm_tan_shp = None

    def jit(self):
        return StandardHamiltonian(self._nll.jit(), _compile_joined=True)

    def draw_sample(self, primals, key, from_inverse=False, cg=cg, **cg_kwargs):
        from jax import random
        subkey_nll, subkey_prr = random.split(key, 2)
        if from_inverse:
            nll_smpl = self._nll.draw_sample(primals, key=subkey_nll)
            prr_inv_metric_smpl = random_like(primals, key=subkey_prr)
            # One may transform any metric sample to a sample of the inverse
            # metric by simply applying the inverse metric to it
            prr_smpl = prr_inv_metric_smpl

            # Note, we can sample antithetically by swapping the global sign of
            # the metric sample below (which corresponds to mirroring the final
            # sample) and additionally by swapping the relative sign between the
            # prior and the likelihood sample. The first technique is
            # computationally cheap and empirically known to improve stability.
            # The latter technique requires an additional inversion and its
            # impact on stability is still unknown.
            # TODO: investigate the impact of sampling the prior and likelihood
            # antithetically.
            met_smpl = nll_smpl + prr_smpl
            # TODO: Set sensible convergence criteria
            """
            lambda x: met(pos, x),
            absdelta=1. / 100,
            resnorm=np.linalg.norm(met_smpl, ord=1) / 2,
            norm_ord=1
            """
            signal_smpl = self.inv_metric(
                primals, met_smpl, cg=cg, x0=prr_inv_metric_smpl, **cg_kwargs
            )
            return signal_smpl
        else:
            nll_smpl = self._nll.draw_sample(primals, key=subkey_nll)
            prr_inv_metric_smpl = random_like(primals, key=subkey_prr)
            return nll_smpl + prr_smpl, key


def Gaussian(data, noise_cov_inv=None, noise_std_inv=None):

    if not noise_cov_inv and not noise_std_inv:

        def noise_cov_inv(tangents):
            return tangents

        def noise_std_inv(tangents):
            return tangents
    elif not noise_std_inv:
        wm = (
            "assuming a diagonal covariance matrix"
            ";\nsetting `noise_std_inv` to"
            " `noise_cov_inv(np.ones_like(data))**0.5`"
        )
        import sys
        print(wm, file=sys.stderr)
        noise_cov_inv_sqrt = np.sqrt(
            noise_cov_inv(tree_map(np.ones_like, data))
        )

        def noise_std_inv(tangents):
            return noise_cov_inv_sqrt * tangents

    def hamiltonian(primals):
        p_res = primals - data
        return 0.5 * np.sum(p_res * noise_cov_inv(p_res))

    def metric(primals, tangents):
        return noise_cov_inv(tangents)

    def left_sqrt_metric(primals, tangents):
        return noise_std_inv(tangents)

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, data)

    return Likelihood(
        hamiltonian,
        metric=metric,
        left_sqrt_metric=left_sqrt_metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


def Categorical(data, axis=-1):
    """
    Provides a categorical likelihood of the data, equivalent to cross entropy

    Parameters
    ----------
    data: np.array(int)
    An array stating which of the categories is the realized in the data
    Must agree with the input shape except for the shape[axis]

    axis: int
    axis over which the categories are formed
    """
    def hamiltonian(primals):
        from jax.nn import log_softmax
        logits = log_softmax(primals, axis=axis)
        return -np.sum(np.take_along_axis(logits, data, axis))

    def metric(primals, tangents):
        from jax.nn import softmax

        preds = softmax(primals, axis=axis)
        norm_term = np.sum(preds * tangents, axis=axis, keepdims=True)
        return preds * tangents - preds * norm_term

    def left_sqrt_metric(primals, tangents):
        from jax.nn import softmax

        sqrtp = np.sqrt(softmax(primals, axis=axis))
        norm_term = np.sum(sqrtp * tangents, axis=axis, keepdims=True)
        return sqrtp * (tangents - sqrtp * norm_term)

    lsm_tangents_shape = tree_map(ShapeWithDtype.from_leave, data)

    return Likelihood(
        hamiltonian,
        metric=metric,
        left_sqrt_metric=left_sqrt_metric,
        lsm_tangents_shape=lsm_tangents_shape
    )
