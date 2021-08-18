from typing import Callable, Optional

from jax import random
from jax.tree_util import Partial

from .optimize import cg
from .sugar import random_like, random_like_shapewdtype, mean
from .likelihood import Likelihood, StandardHamiltonian


def sample_likelihood(likelihood: Likelihood, primals, key):
    white_sample = random_like_shapewdtype(
        likelihood.left_sqrt_metric_tangents_shape, key=key
    )
    return likelihood.left_sqrt_metric(primals, white_sample)


def _sample_standard_hamiltonian(
    hamiltonian: StandardHamiltonian,
    primals,
    key,
    from_inverse: bool,
    cg: Callable = cg,
    cg_kwargs: Optional[dict] = None
):
    if not isinstance(hamiltonian, StandardHamiltonian):
        te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
        raise TypeError(te)
    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    subkey_nll, subkey_prr = random.split(key, 2)
    nll_smpl = sample_likelihood(
        hamiltonian.likelihood, primals, key=subkey_nll
    )
    prr_inv_metric_smpl = random_like(primals, key=subkey_prr)
    # One may transform any metric sample to a sample of the inverse
    # metric by simply applying the inverse metric to it
    prr_smpl = prr_inv_metric_smpl
    # Note, we can sample antithetically by swapping the global sign of
    # the metric sample below (which corresponds to mirroring the final
    # sample) and additionally by swapping the relative sign between
    # the prior and the likelihood sample. The first technique is
    # computationally cheap and empirically known to improve stability.
    # The latter technique requires an additional inversion and its
    # impact on stability is still unknown.
    # TODO: investigate the impact of sampling the prior and likelihood
    # antithetically.
    met_smpl = nll_smpl + prr_smpl
    if from_inverse:
        # TODO: Set sensible convergence criteria
        signal_smpl = hamiltonian.inv_metric(
            primals, met_smpl, cg=cg, x0=prr_inv_metric_smpl, **cg_kwargs
        )
        return signal_smpl, met_smpl
    else:
        return None, met_smpl


def sample_standard_hamiltonian(*args, **kwargs):
    r"""Draws a sample of which the covariance is the metric or the inverse
    metric of the Hamiltonian.

    To sample from the inverse metric, we need to be able to draw samples
    which have the metric as covariance structure and we need to be able to
    apply the inverse metric. The first part is trivial since we can use
    the left square root of the metric :math:`L` associated with every
    likelihood:

    .. math::
        :nowrap:

        \begin{gather*}
            \tilde{d} \leftarrow \mathcal{G}(0,\mathbb{1}) \\
            t = L \tilde{d}
        \end{gather*}

    with :math:`t` now having a covariance structure of

    .. math::
        <t t^\dagger> = L <\tilde{d} \tilde{d}^\dagger> L^\dagger = M .

    We now need to apply the inverse metric in order to transform the
    sample to an inverse sample. We can do so using the conjugate gradient
    algorithm which yields the solution to $M s = t$, i.e. applies the
    inverse of $M$ to $t$:

    .. math::
        :nowrap:

        \begin{gather*}
            M s =  t \\
            s = M^{-1} t = cg(M, t) .
        \end{gather*}

    Parameters
    ----------
    hamiltonian: StandardHamiltonian
        Hamiltonian with standard prior from which to draw samples.
    primals : tree-like structure
        Position at which to draw samples.
    key : tuple, list or np.ndarray of uint32 of length two
        Random key with which to generate random variables in data domain.
    cg : callable, optional
        Implementation of the conjugate gradient algorithm and used to
        apply the inverse of the metric.
    cg_kwargs : dict, optional
        Additional keyword arguments passed on to `cg`.

    Returns
    -------
    sample : tree-like structure
        Sample of which the covariance is the inverse metric.
    """
    inv_met_smpl, _ = _sample_standard_hamiltonian(
        *args, from_inverse=True, **kwargs
    )
    return inv_met_smpl


def geometrically_sample_standard_hamiltonian(
    hamiltonian: StandardHamiltonian,
    primals,
    key,
    mirror_linear_sample: bool,
    linear_sampling_cg: Callable = cg,
    linear_sampling_kwargs: Optional[dict] = None,
    non_linear_sampling_kwargs: Optional[dict] = None
):
    r"""Draws a sample which follows a standard normal distribution in the
    canonical coordinate system of the Riemannian manifold associated with the
    metric of the other distribution. The coordinate transformation is
    approximated by expanding around a given point `primals`.

    Parameters
    ----------
    hamiltonian: StandardHamiltonian
        Hamiltonian with standard prior from which to draw samples.
    primals : tree-like structure
        Position at which to draw samples.
    key : tuple, list or np.ndarray of uint32 of length two
        Random key with which to generate random variables in data domain.
    linear_sampling_cg : callable
        Implementation of the conjugate gradient algorithm and used to
        apply the inverse of the metric.
    linear_sampling_kwargs : dict
        Additional keyword arguments passed on to `cg`.
    non_linear_sampling_kwargs : dict
        Additional keyword arguments passed on to the minimzer of the
        non-linear potential.

    Returns
    -------
    sample : tree-like structure
        Sample of which the covariance is the inverse metric.
    """
    if not isinstance(hamiltonian, StandardHamiltonian):
        te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
        raise TypeError(te)
    from . import minimize

    # TODO: @PhilippF Does it make sense to not start from a MGVI sample?
    inv_met_smpl, met_smpl = _sample_standard_hamiltonian(
        hamiltonian,
        primals,
        key=key,
        from_inverse=True,
        cg=linear_sampling_cg,
        cg_kwargs=linear_sampling_kwargs
    )

    nls_kwargs = non_linear_sampling_kwargs
    nls_kwargs = nls_kwargs if nls_kwargs is not None else {}
    # Abort early if non-linear sampling is effectively disabled
    if nls_kwargs.get("maxiter") == 0:
        if mirror_linear_sample:
            return (inv_met_smpl, -inv_met_smpl)
        return (inv_met_smpl, )

    def draw_non_linear_sample(lh, met_smpl, inv_met_smpl):
        x0 = primals + inv_met_smpl

        def fun(x):
            g = x - primals + lh.left_sqrt_metric(
                primals,
                lh.transformation(x) - lh.transformation(primals)
            )
            r = met_smpl - g
            return r.dot(r) / 2.

        core_nls_kwargs = {
            # TODO: @PhilippF Is this really the correct metric?
            "hessp": hamiltonian.metric,
            # TODO: set sensible default convergence criteria
        }

        opt_state = minimize(
            fun, x0=x0, method="NewtonCG", options=core_nls_kwargs | nls_kwargs
        )

        return opt_state

    opt_state_smpl1 = draw_non_linear_sample(
        hamiltonian.likelihood, met_smpl, inv_met_smpl
    )
    if not mirror_linear_sample:
        return (opt_state_smpl1.x - primals, )
    opt_state_smpl2 = draw_non_linear_sample(
        hamiltonian.likelihood, -met_smpl, -inv_met_smpl
    )
    return (opt_state_smpl1.x - primals, opt_state_smpl2.x - primals)


class MetricKL():
    def __init__(
        self,
        hamiltonian,
        primals,
        n_samples,
        key,
        mirror_samples: bool = True,
        cg: Callable = cg,
        cg_kwargs: Optional[dict] = None,
        hamiltonian_and_gradient: Optional[Callable] = None,
        _samples: Optional[tuple] = None
    ):
        if not isinstance(hamiltonian, StandardHamiltonian):
            te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
            raise TypeError(te)
        self._ham = hamiltonian
        self._pos = primals
        self._n_samples = n_samples

        if _samples is None:
            _samples = []
            draw = Partial(
                sample_standard_hamiltonian,
                hamiltonian=hamiltonian,
                primals=primals,
                cg=cg,
                cg_kwargs=cg_kwargs
            )
            subkeys = random.split(key, n_samples)
            _samples = [draw(key=k) for k in subkeys]
            _samples += [-s for s in _samples] if mirror_samples else []
        self._samples = tuple(_samples)

        self._ham_vg = hamiltonian_and_gradient

    def __call__(self, primals):
        return self.energy(primals)

    def energy(self, primals):
        return mean(tuple(self._ham(primals + s) for s in self.samples))

    def energy_and_gradient(self, primals):
        if self._ham_vg is None:
            nie = "need to set `hamiltonian_and_gradient` first"
            raise NotImplementedError(nie)
        # gradient of mean is the mean of the gradients
        return mean(tuple(self._ham_vg(primals + s) for s in self.samples))

    def metric(self, primals, tangents):
        return mean(
            tuple(
                self._ham.metric(primals + s, tangents) for s in self.samples
            )
        )

    @property
    def position(self):
        return self._pos

    @property
    def n_samples(self):
        return self.n_samples

    @property
    def samples(self):
        return tuple(self._samples)


class GeoMetricKL(MetricKL):
    def __init__(
        self,
        hamiltonian: StandardHamiltonian,
        primals,
        n_samples,
        key,
        mirror_samples: bool = True,
        linear_sampling_cg: Callable = cg,
        linear_sampling_kwargs: Optional[dict] = None,
        non_linear_sampling_kwargs: Optional[dict] = None,
        hamiltonian_and_gradient: Optional[Callable] = None,
        _samples: Optional[tuple] = None
    ):
        if not isinstance(hamiltonian, StandardHamiltonian):
            te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
            raise TypeError(te)
        self._ham = hamiltonian
        self._pos = primals
        self._n_samples = n_samples

        if _samples is None:
            draw = Partial(
                geometrically_sample_standard_hamiltonian,
                hamiltonian=hamiltonian,
                primals=primals,
                mirror_linear_sample=mirror_samples,
                linear_sampling_cg=linear_sampling_cg,
                linear_sampling_kwargs=linear_sampling_kwargs,
                non_linear_sampling_kwargs=non_linear_sampling_kwargs
            )
            subkeys = random.split(key, n_samples)
            _samples = tuple(
                s for smpl_tuple in (draw(key=k) for k in subkeys)
                for s in smpl_tuple
            )
        self._samples = tuple(_samples)

        self._ham_vg = hamiltonian_and_gradient
