from jax import numpy as np
from jax.scipy.sparse.linalg import cg #TODO: replace
from .operator import Likelihood
from .sugar import makeField, just_add

class StandardHamiltonian(Likelihood):
    def __init__(
        self, likelihood
    ):
        """
        Adds a standard normal prior to a likelihood and flattens any data structures
        TODO: actually flatten the pytree
        """
        self._nll = likelihood

        def joined_hamiltonian(primals):
            return self._nll(primals) + makeField(primals).norm()

        def joined_metric(primals, tangents):
            return just_add(self._nll.metric(primals, tangents), tangents)

        self._hamiltonian = joined_hamiltonian
        self._metric = joined_metric
        self._draw_metric_sample = None

    def draw_sample(
        self,
        primals,
        key,
        from_inverse = False,
        x0 = None,
        maxiter = None,
        **kwargs
    ):
        from jax import random
        key, subkey_nll, subkey_prr = random.split(key, 3)
        if from_inverse:
            nll_smpl, _ = self._nll.draw_sample(primals, key=subkey_nll, **kwargs)
            prr_inv_metric_smpl = random.normal(shape=primals.shape,
                    key=subkey_prr)
            # Shorthand for retrieving the sample from an inverse sample
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
            met_smpl = just_add(nll_smpl, prr_smpl)
            signal_smpl = cg(
                lambda t, primals=primals: self._metric(primals, t),
                met_smpl,
                x0=prr_inv_metric_smpl if x0 is None else x0,
                maxiter=maxiter
            )[0]
            return signal_smpl, key
        else:
            nll_smpl, _ = self._nll.draw_sample(primals, key=subkey_nll, **kwargs)
            prr_inv_metric_smpl = random.normal(shape=primals.shape,
                    key=subkey_prr)
            return just_add(nll_smpl,prr_smpl), key


def Gaussian(
    data,
    noise_cov_inv = None,
    noise_std_inv = None):

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
        noise_cov_inv_sqrt = np.sqrt(noise_cov_inv(np.ones_like(data)))

        def noise_std_inv(tangents):
            return noise_cov_inv_sqrt * tangents

    def energy(primals):
        p_res = primals - data
        return 0.5 * np.sum(p_res * noise_cov_inv(p_res))

    def metric(primals, tangents):
        return noise_cov_inv(tangents)

    def draw_sample(primals, key):
        from jax import random

        key, subkey = random.split(key)
        tangents = random.normal(shape=data.shape, key=subkey)
        return noise_std_inv(tangents), key

    return Likelihood(energy, metric, draw_sample)


