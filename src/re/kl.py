# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

import jax
from jax import lax
from jax import random
from jax.tree_util import Partial, register_pytree_node_class

from . import conjugate_gradient
from .forest_util import assert_arithmetics, map_forest, map_forest_mean, unstack
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


def _sample_standard_hamiltonian(
    hamiltonian: StandardHamiltonian,
    primals,
    key,
    from_inverse: bool,
    cg: Callable = conjugate_gradient.static_cg,
    cg_name: Optional[str] = None,
    cg_kwargs: Optional[dict] = None,
):
    if not isinstance(hamiltonian, StandardHamiltonian):
        te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
        raise TypeError(te)
    cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    subkey_nll, subkey_prr = random.split(key, 2)
    nll_smpl = sample_likelihood(
        hamiltonian.likelihood, primals, key=subkey_nll
    )
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
            cg, Partial(hamiltonian.metric, primals), **{
                "name": cg_name,
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
    hamiltonian: StandardHamiltonian, primals, *args, **kwargs
):
    r"""Draws a sample of which the covariance is the metric or the inverse
    metric of the Hamiltonian.

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
    hamiltonian:
        Hamiltonian with standard prior from which to draw samples.
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
        hamiltonian, primals, *args, from_inverse=True, **kwargs
    )
    return inv_met_smpl


def geometrically_sample_standard_hamiltonian(
    hamiltonian: StandardHamiltonian,
    primals,
    key,
    mirror_linear_sample: bool,
    linear_sampling_cg: Callable = conjugate_gradient.static_cg,
    linear_sampling_name: Optional[str] = None,
    linear_sampling_kwargs: Optional[dict] = None,
    non_linear_sampling_method: str = "NewtonCG",
    non_linear_sampling_name: Optional[str] = None,
    non_linear_sampling_kwargs: Optional[dict] = None,
):
    r"""Draws a sample which follows a standard normal distribution in the
    canonical coordinate system of the Riemannian manifold associated with the
    metric of the other distribution. The coordinate transformation is
    approximated by expanding around a given point `primals`.

    Parameters
    ----------
    hamiltonian:
        Hamiltonian with standard prior from which to draw samples.
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
    if not isinstance(hamiltonian, StandardHamiltonian):
        te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
        raise TypeError(te)
    assert_arithmetics(primals)
    from .energy_operators import Gaussian
    from .optimize import minimize

    inv_met_smpl, met_smpl = _sample_standard_hamiltonian(
        hamiltonian,
        primals,
        key=key,
        from_inverse=True,
        cg=linear_sampling_cg,
        cg_name=linear_sampling_name,
        cg_kwargs=linear_sampling_kwargs
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

    lh_trafo_at_p = hamiltonian.likelihood.transformation(primals)

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

    smpl1, smpl1_status = draw_non_linear_sample(
        hamiltonian.likelihood, met_smpl, inv_met_smpl
    )
    cond_raise(
        (smpl1_status is not None) & (smpl1_status < 0),
        ValueError("S: failed to invert map")
    )
    if not mirror_linear_sample:
        return (smpl1 - primals, )
    smpl2, smpl2_status = draw_non_linear_sample(
        hamiltonian.likelihood, -met_smpl, -inv_met_smpl
    )
    cond_raise(
        (smpl2_status is not None) & (smpl2_status < 0),
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
    hamiltonian: StandardHamiltonian,
    primals,
    n_samples: int,
    key,
    mirror_samples: bool = True,
    sample_mapping: Union[str, Callable] = 'lax',
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

    hamiltonian : :class:`nifty8.src.re.likelihood.StandardHamiltonian`
        Hamiltonian of the approximated probability distribution.
    primals : :class:`nifty8.re.field.Field`
        Expansion point of the coordinate transformation.
    n_samples : integer
        Number of samples used to stochastically estimate the KL.
    key : DeviceArray
        A PRNG-key.
    mirror_samples : boolean
        Whether the mirrored version of the drawn samples are also used.
        If true, the number of used samples doubles.
        Mirroring samples stabilizes the KL estimate as extreme
        sample variation is counterbalanced.
        Default is True.
    sample_mapping : string, callable
        Can be either a string-key to a mapping function or a mapping function
        itself. The function is used to map the drawing of samples. Possible
        string-keys are:

        keys                -       functions
        -------------------------------------
        'pmap' or 'p'       -       jax.pmap
        'lax.map' or 'lax'  -       jax.lax.map

        In case sample_mapping is passed as a function, it should produce a
        mapped function f_mapped of a general function f as: `f_mapped =
        sample_mapping(f)`
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
    if not isinstance(hamiltonian, StandardHamiltonian):
        te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
        raise TypeError(te)
    assert_arithmetics(primals)

    draw = partial(
        sample_standard_hamiltonian,
        hamiltonian=hamiltonian,
        primals=primals,
        cg=linear_sampling_cg,
        cg_name=linear_sampling_name,
        cg_kwargs=linear_sampling_kwargs
    )
    subkeys = random.split(key, n_samples)
    if isinstance(sample_mapping, str):
        if sample_mapping == 'pmap' or sample_mapping == 'p':
            sample_mapping = jax.pmap
        elif sample_mapping == 'lax.map' or sample_mapping == 'lax':
            sample_mapping = partial(partial, lax.map)
        else:
            ve = (
                f"{sample_mapping} is not an accepted key to a mapping function"
                "; please pass function directly"
            )
            raise ValueError(ve)

    elif not callable(sample_mapping):
        te = (
            f"invalid `sample_mapping` of type {type(sample_mapping)!r}"
            "; expected string or callable"
        )
        raise TypeError(te)

    samples_stack = sample_mapping(lambda k: draw(key=k))(subkeys)

    return SampleIter(
        mean=primals,
        samples=unstack(samples_stack),
        linearly_mirror_samples=mirror_samples
    )


def GeoMetricKL(
    hamiltonian: StandardHamiltonian,
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
    if not isinstance(hamiltonian, StandardHamiltonian):
        te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
        raise TypeError(te)
    assert_arithmetics(primals)

    draw = partial(
        geometrically_sample_standard_hamiltonian,
        hamiltonian=hamiltonian,
        primals=primals,
        mirror_linear_sample=mirror_samples,
        linear_sampling_cg=linear_sampling_cg,
        linear_sampling_name=linear_sampling_name,
        linear_sampling_kwargs=linear_sampling_kwargs,
        non_linear_sampling_method=non_linear_sampling_method,
        non_linear_sampling_name=non_linear_sampling_name,
        non_linear_sampling_kwargs=non_linear_sampling_kwargs
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


def mean_value_and_grad(ham: Callable, sample_mapping='vmap', *args, **kwargs):
    """Thin wrapper around `value_and_grad` and the provided sample mapping
    function, e.g. `vmap` to apply a cost function to a mean and a list of
    residual samples.

    Parameters
    ----------

    ham : :class:`nifty8.src.re.likelihood.StandardHamiltonian`
        Hamiltonian of the approximated probability distribution,
        of which the mean value and the mean gradient are to be computed.
    sample_mapping : string, callable
        Can be either a string-key to a mapping function or a mapping function
        itself. The function is used to map the drawing of samples. Possible
        string-keys are:

        keys                -       functions
        -------------------------------------
        'vmap' or 'v'       -       jax.vmap
        'pmap' or 'p'       -       jax.pmap
        'lax.map' or 'lax'  -       jax.lax.map

        In case sample_mapping is passed as a function, it should produce a
        mapped function f_mapped of a general function f as: `f_mapped =
        sample_mapping(f)`
    """
    from jax import value_and_grad
    vg = value_and_grad(ham, *args, **kwargs)

    def mean_vg(
        primals: P,
        primals_samples: Union[None, Sequence[P], SampleIter] = None,
        **primals_kw
    ) -> Tuple[Any, P]:
        ham_vg = partial(vg, **primals_kw)
        if primals_samples is None:
            return ham_vg(primals)

        if not isinstance(primals_samples, SampleIter):
            primals_samples = SampleIter(samples=primals_samples)
        return map_forest_mean(ham_vg, mapping=sample_mapping, in_axes=(0, ))(
            tuple(primals_samples.at(primals))
        )

    return mean_vg


def mean_hessp(ham: Callable, *args, **kwargs):
    """Thin wrapper around `jvp`, `grad` and `vmap` to apply a binary method to
    a primal mean, a tangent and a list of residual primal samples.
    """
    from jax import jvp, grad
    jac = grad(ham, *args, **kwargs)

    def mean_hp(
        primals: P,
        tangents: Any,
        primals_samples: Union[None, Sequence[P], SampleIter] = None,
        **primals_kw
    ) -> P:
        if primals_samples is None:
            _, hp = jvp(partial(jac, **primals_kw), (primals, ), (tangents, ))
            return hp

        if not isinstance(primals_samples, SampleIter):
            primals_samples = SampleIter(samples=primals_samples)
        return map_forest_mean(
            partial(mean_hp, primals_samples=None, **primals_kw),
            in_axes=(0, None)
        )(tuple(primals_samples.at(primals)), tangents)

    return mean_hp


def mean_metric(metric: Callable):
    """Thin wrapper around `vmap` to apply a binary method to a primal mean, a
    tangent and a list of residual primal samples.
    """
    def mean_met(
        primals: P,
        tangents: Any,
        primals_samples: Union[None, Sequence[P], SampleIter] = None,
        **primals_kw
    ) -> P:
        if primals_samples is None:
            return metric(primals, tangents, **primals_kw)

        if not isinstance(primals_samples, SampleIter):
            primals_samples = SampleIter(samples=primals_samples)
        return map_forest_mean(
            partial(metric, **primals_kw), in_axes=(0, None)
        )(tuple(primals_samples.at(primals)), tangents)

    return mean_met
