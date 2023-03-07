# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Callable, Optional, Sequence, TypeVar, Union
from warnings import warn

from jax import random
from jax.tree_util import Partial, register_pytree_node_class

from . import conjugate_gradient
from .forest_util import (
    assert_arithmetics,
    get_map,
    map_forest,
    map_forest_mean,
    unstack,
)
from .likelihood import Likelihood, StandardHamiltonian
from .sugar import random_like

P = TypeVar("P")


def sample_likelihood(likelihood: Likelihood, primals, key):
    white_sample = random_like(key, likelihood.left_sqrt_metric_tangents_shape)
    return likelihood.left_sqrt_metric(primals, white_sample)


def cond_raise(condition, exception):
    from jax.experimental.host_callback import call

    def maybe_raise(condition):
        if condition:
            raise exception

    call(maybe_raise, condition, result_shape=None)


def _likelihood_metric_plus_standard_prior(lh_metric):
    if isinstance(lh_metric, Likelihood):
        lh_metric = lh_metric.metric

    def joined_metric(primals, tangents, **primals_kw):
        return lh_metric(primals, tangents, **primals_kw) + tangents

    return joined_metric


def _sample_standard_hamiltonian(
    likelihood: Likelihood,
    primals,
    key,
    from_inverse: bool,
    cg: Callable = conjugate_gradient.static_cg,
    cg_name: Optional[str] = None,
    cg_kwargs: Optional[dict] = None,
    _raise_nonposdef: bool = False,
):
    if isinstance(likelihood, Likelihood):
        lh = likelihood
        ham_metric = _likelihood_metric_plus_standard_prior(lh)
    elif isinstance(likelihood, StandardHamiltonian):
        msg = "passing `StandardHamiltonian` instead of the `Likelihood` is deprecated"
        warn(msg, DeprecationWarning)
        lh = likelihood.likelihood
        ham_metric = likelihood.metric
    else:
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    subkey_nll, subkey_prr = random.split(key, 2)
    nll_smpl = sample_likelihood(lh, primals, key=subkey_nll)
    prr_inv_metric_smpl = random_like(key=subkey_prr, primals=primals)
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
        inv_metric_at_p = partial(
            cg, Partial(ham_metric, primals), **{
                "name": cg_name,
                "_raise_nonposdef": _raise_nonposdef,
                **cg_kwargs
            }
        )
        signal_smpl, info = inv_metric_at_p(met_smpl, x0=prr_inv_metric_smpl)
        cond_raise(
            (info is not None) & (info < 0),
            ValueError("conjugate gradient failed")
        )
        return signal_smpl, met_smpl
    else:
        return None, met_smpl


def sample_standard_hamiltonian(
    likelihood: Likelihood, primals, *args, **kwargs
):
    r"""Draws a sample of which the covariance is the metric or the inverse
    metric of the likelihood with assumed standard normal prior.

    To sample from the inverse metric, we need to be able to draw samples
    which have the metric as covariance structure and we need to be able to
    apply the inverse metric. The first part is trivial since we can use
    the left square root of the metric :math:`L` associated with every
    likelihood:

    .. math::

        \tilde{d} \leftarrow \mathcal{G}(0,\mathbb{1}) \\
        t = L \tilde{d}

    with :math:`t` now having a covariance structure of

    .. math::
        <t t^\dagger> = L <\tilde{d} \tilde{d}^\dagger> L^\dagger = M .

    We now need to apply the inverse metric in order to transform the
    sample to an inverse sample. We can do so using the conjugate gradient
    algorithm which yields the solution to :math:`M s = t`, i.e. applies the
    inverse of :math:`M` to :math:`t`:

    .. math::

        M &s =  t \\
        &s = M^{-1} t = cg(M, t) .

    Parameters
    ----------
    likelihood:
        Likelihood with assumed standard prior from which to draw samples.
    primals : tree-like structure
        Position at which to draw samples.
    key : tuple, list or jnp.ndarray of uint32 of length two
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

    See also
    --------
    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """
    assert_arithmetics(primals)
    inv_met_smpl, _ = _sample_standard_hamiltonian(
        likelihood, primals, *args, from_inverse=True, **kwargs
    )
    return inv_met_smpl


# TODO (?): optionally accept ham.metric and likelihood.lsm and likelihood.transformation
def geometrically_sample_standard_hamiltonian(
    likelihood: Likelihood,
    primals,
    key,
    mirror_linear_sample: bool,
    linear_sampling_cg: Callable = conjugate_gradient.static_cg,
    linear_sampling_name: Optional[str] = None,
    linear_sampling_kwargs: Optional[dict] = None,
    non_linear_sampling_method: str = "NewtonCG",
    non_linear_sampling_name: Optional[str] = None,
    non_linear_sampling_kwargs: Optional[dict] = None,
    _raise_notconverged: bool = False,
):
    r"""Draws a sample which follows a standard normal distribution in the
    canonical coordinate system of the Riemannian manifold associated with the
    metric of the other distribution. The coordinate transformation is
    approximated by expanding around a given point `primals`.

    Parameters
    ----------
    likelihood:
        Likelihood with assumed standard prior from which to draw samples.
    primals : tree-like structure
        Position at which to draw samples.
    key : tuple, list or jnp.ndarray of uint32 of length two
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

    See also
    --------
    `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
    Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
    `<https://doi.org/10.3390/e23070853>`_
    """
    from .energy_operators import Gaussian
    from .optimize import minimize

    if isinstance(likelihood, Likelihood):
        lh = likelihood
    elif isinstance(likelihood, StandardHamiltonian):
        msg = "passing the StandardHamiltonian instead of the Likelihood is deprecated"
        warn(msg, DeprecationWarning)
        lh = likelihood.likelihood
    else:
        te = f"`likelihood` of invalid type; got '{type(likelihood)}'"
        raise TypeError(te)
    assert_arithmetics(primals)

    inv_met_smpl, met_smpl = _sample_standard_hamiltonian(
        likelihood,
        primals,
        key=key,
        from_inverse=True,
        cg=linear_sampling_cg,
        cg_name=linear_sampling_name,
        cg_kwargs=linear_sampling_kwargs,
        _raise_nonposdef=_raise_notconverged,
    )

    if isinstance(non_linear_sampling_kwargs, dict):
        nls_kwargs = non_linear_sampling_kwargs
    elif non_linear_sampling_kwargs is None:
        nls_kwargs = {}
    else:
        te = (
            "`non_linear_sampling_kwargs` of invalid type"
            "{type(non_linear_sampling_kwargs)}"
        )
        raise TypeError(te)
    nls_kwargs = {"name": non_linear_sampling_name, **nls_kwargs}
    if "hessp" in nls_kwargs:
        ve = "setting the hessian for an unknown function is invalid"
        raise ValueError(ve)
    # Abort early if non-linear sampling is effectively disabled
    if nls_kwargs.get("maxiter") == 0:
        if mirror_linear_sample:
            return (inv_met_smpl, -inv_met_smpl)
        return (inv_met_smpl, )

    lh_trafo_at_p = lh.transformation(primals)

    def draw_non_linear_sample(lh, met_smpl, inv_met_smpl):
        x0 = primals + inv_met_smpl

        def g(x):
            return x - primals + lh.left_sqrt_metric(
                primals,
                lh.transformation(x) - lh_trafo_at_p
            )

        r2_half = Gaussian(met_smpl) @ g  # (g - met_smpl)**2 / 2

        options = nls_kwargs.copy()
        options["hessp"] = r2_half.metric

        opt_state = minimize(
            r2_half, x0=x0, method=non_linear_sampling_method, options=options
        )

        return opt_state.x, opt_state.status

    smpl1, smpl1_status = draw_non_linear_sample(lh, met_smpl, inv_met_smpl)
    cond_raise(
        _raise_notconverged & (smpl1_status is not None) & (smpl1_status < 0),
        ValueError("S: failed to invert map")
    )
    if not mirror_linear_sample:
        return (smpl1 - primals, )
    smpl2, smpl2_status = draw_non_linear_sample(lh, -met_smpl, -inv_met_smpl)
    cond_raise(
        _raise_notconverged & (smpl2_status is not None) & (smpl2_status < 0),
        ValueError("S: failed to invert map")
    )
    return (smpl1 - primals, smpl2 - primals)


@register_pytree_node_class
class SampleIter():
    """Storage class for samples with some convenience methods for applying
    operators of them

    This class is used to store samples for the Variational Inference schemes
    MGVI and geoVI where samples are defined relative to some expansion point
    (a.k.a. latent mean or offset).

    See also
    --------
    `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
    Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
    `<https://doi.org/10.3390/e23070853>`_

    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """
    def __init__(
        self,
        *,
        mean: P = None,
        samples: Sequence[P],
        linearly_mirror_samples: bool = False,
    ):
        self._samples = tuple(samples)
        self._mean = mean

        self._n_samples = len(self._samples)
        if linearly_mirror_samples == True:
            self._n_samples *= 2
        self._linearly_mirror_samples = linearly_mirror_samples
        # TODO/IDEA: Implement a transposed SampleIter object (SampleStack)
        # akin to `vmap_forest_mean`

    def __iter__(self):
        for s in self._samples:
            yield self._mean + s if self._mean is not None else s
            if self._linearly_mirror_samples:
                yield self._mean - s if self._mean is not None else -s

    def __len__(self):
        return self._n_samples

    @property
    def n_samples(self):
        """Total number of samples, equivalent to the length of the object"""
        return len(self)

    def at(self, mean):
        """Updates the offset (usually the latent mean) of all samples"""
        return SampleIter(
            mean=mean,
            samples=self._samples,
            linearly_mirror_samples=self._linearly_mirror_samples
        )

    @property
    def first(self):
        """Convenience method to easily retrieve a sample (the first one)"""
        if self._mean is not None:
            return self._mean + self._samples[0]
        return self._samples[0]

    def apply(self, call: Callable, *args, **kwargs):
        """Applies an operator over all samples, yielding a list of outputs

        Internally, the call is `vmap`-ed over the samples for additional
        efficiency.
        """
        if set(kwargs.keys()) | {"in_axes"} != {"in_axes"}:
            raise ValueError(f"invalid keyword arguments {kwargs}")

        # TODO: vmap is significantly slower than looping over the samples
        # for an extremely high dimensional problem.
        in_axes = kwargs.get("in_axes", (0, ))
        return map_forest(call, in_axes=in_axes)(tuple(self), *args)

    def mean(self, call: Callable, *args, **kwargs):
        """Applies an operator over all samples and averages the results

        Internally, the call is `vmap`-ed over the samples for additional
        efficiency.
        """
        if set(kwargs.keys()) | {"in_axes"} != {"in_axes"}:
            raise ValueError(f"invalid keyword arguments {kwargs}")

        # TODO: vmap is significantly slower than looping over the samples
        # for an extremely high dimensional problem.
        in_axes = kwargs.get("in_axes", (0, ))
        return map_forest_mean(call, in_axes=in_axes)(tuple(self), *args)

    def tree_flatten(self):
        return ((self._mean, self._samples), (self._linearly_mirror_samples, ))

    @classmethod
    def tree_unflatten(cls, aux, children):
        if len(aux) != 1 or len(children) != 2:
            raise ValueError()
        return cls(
            mean=children[0],
            samples=children[1],
            linearly_mirror_samples=aux[0]
        )


def MetricKL(
    likelihood: Likelihood,
    primals,
    n_samples: int,
    key,
    mirror_samples: bool = True,
    map: Union[str, Callable] = 'lax',
    linear_sampling_cg: Callable = conjugate_gradient.static_cg,
    linear_sampling_name: Optional[str] = None,
    linear_sampling_kwargs: Optional[dict] = None,
) -> SampleIter:
    """Provides the sampled Kullback-Leibler divergence between a distribution
    and a Metric Gaussian.

    A Metric Gaussian is used to approximate another probability distribution.
    It is a Gaussian distribution that uses the Fisher information metric of
    the other distribution at the location of its mean to approximate the
    variance. In order to infer the mean, a stochastic estimate of the
    Kullback-Leibler divergence is minimized. This estimate is obtained by
    sampling the Metric Gaussian at the current mean. During minimization these
    samples are kept constant and only the mean is updated. Due to the
    typically nonlinear structure of the true distribution these samples have
    to be updated eventually by re-instantiating the Metric Gaussian again. For
    the true probability distribution the standard parametrization is assumed.

    Parameters
    ----------
    likelihood :
        Likelihood with assumed standard prior for which the probability
        distribution is approximated.
    primals : :class:`nifty8.re.field.Field`
        Expansion point of the coordinate transformation.
    n_samples : integer
        Number of samples used to stochastically estimate the KL.
    key : DeviceArray
        A PRNG-key.
    mirror_samples : bool
        Whether the mirrored version of the drawn samples are also used.
        If true, the number of used samples doubles.
        Mirroring samples stabilizes the KL estimate as extreme
        sample variation is counterbalanced.
        Default is True.
    map : string, callable
        Can be either a string-key to a mapping function or a mapping function
        itself. The function is used to map the drawing of samples. Possible
        string-keys are:

        - 'pmap' or 'p' for `jax.pmap`
        - 'lax.map' or 'lax' for `jax.lax.map`

        In case `map` is passed as a function, it should produce a mapped
        function f_mapped of a general function f as: `f_mapped = map(f)`.
    linear_sampling_cg : callable
        Implementation of the conjugate gradient algorithm and used to
        apply the inverse of the metric.
    linear_sampling_name : string, optional
        'name'-keyword-argument passed to `linear_sampling_cg`.
    linear_sampling_kwargs : dict, optional
        Additional keyword arguments passed on to `linear_sampling_cg`.

    See also
    --------
    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """
    draw = partial(
        sample_standard_hamiltonian,
        likelihood=likelihood,
        primals=primals,
        cg=linear_sampling_cg,
        cg_name=linear_sampling_name,
        cg_kwargs=linear_sampling_kwargs
    )
    subkeys = random.split(key, n_samples)

    map = get_map(map)
    samples_stack = map(lambda k: draw(key=k))(subkeys)

    return SampleIter(
        mean=primals,
        samples=unstack(samples_stack),
        linearly_mirror_samples=mirror_samples
    )


def GeoMetricKL(
    likelihood: Likelihood,
    primals,
    n_samples: int,
    key,
    mirror_samples: bool = True,
    linear_sampling_cg: Callable = conjugate_gradient.static_cg,
    linear_sampling_name: Optional[str] = None,
    linear_sampling_kwargs: Optional[dict] = None,
    non_linear_sampling_method: str = "NewtonCG",
    non_linear_sampling_name: Optional[str] = None,
    non_linear_sampling_kwargs: Optional[dict] = None,
    _raise_notconverged: bool = False,
) -> SampleIter:
    """Provides the sampled Kullback-Leibler used in geometric Variational
    Inference (geoVI).

    In geoVI a probability distribution is approximated with a standard normal
    distribution in the canonical coordinate system of the Riemannian manifold
    associated with the metric of the other distribution. The coordinate
    transformation is approximated by expanding around a point. In order to
    infer the expansion point, a stochastic estimate of the Kullback-Leibler
    divergence is minimized. This estimate is obtained by sampling from the
    approximation using the current expansion point. During minimization these
    samples are kept constant and only the expansion point is updated. Due to
    the typically nonlinear structure of the true distribution these samples
    have to be updated eventually by re-instantiating the geometric Gaussian
    again. For the true probability distribution the standard parametrization
    is assumed.

    See also
    --------
    `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
    Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
    `<https://doi.org/10.3390/e23070853>`_
    """
    draw = partial(
        geometrically_sample_standard_hamiltonian,
        likelihood=likelihood,
        primals=primals,
        mirror_linear_sample=mirror_samples,
        linear_sampling_cg=linear_sampling_cg,
        linear_sampling_name=linear_sampling_name,
        linear_sampling_kwargs=linear_sampling_kwargs,
        non_linear_sampling_method=non_linear_sampling_method,
        non_linear_sampling_name=non_linear_sampling_name,
        non_linear_sampling_kwargs=non_linear_sampling_kwargs,
        _raise_notconverged=_raise_notconverged,
    )
    subkeys = random.split(key, n_samples)
    # TODO: Make `geometrically_sample_standard_hamiltonian` jit-able
    # samples_stack = lax.map(lambda k: draw(key=k), subkeys)
    # Unpack tuple of samples
    # samples_stack = tree_map(
    #     lambda a: a.reshape((-1, ) + a.shape[2:]), samples_stack
    # )
    # samples = unstack(samples_stack)
    samples = tuple(s for ss in map(lambda k: draw(key=k), subkeys) for s in ss)

    return SampleIter(
        mean=primals, samples=samples, linearly_mirror_samples=False
    )
