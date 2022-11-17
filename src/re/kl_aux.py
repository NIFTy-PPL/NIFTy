# Copyright(C) 2013-2022 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union

from .forest_util import map_forest_mean
from .kl import SampleIter

P = TypeVar("P")


def mean_value_and_grad(ham: Callable, map="vmap", *args, **kwargs):
    """Thin wrapper around `value_and_grad` and the provided sample mapping
    function, e.g. `vmap` to apply a cost function to a mean and a list of
    residual samples.

    Parameters
    ----------
    ham : :class:`nifty8.src.re.likelihood.StandardHamiltonian`
        Hamiltonian of the approximated probability distribution,
        of which the mean value and the mean gradient are to be computed.
    map : string, callable
        Can be either a string-key to a mapping function or a mapping function
        itself. The function is used to map the drawing of samples. Possible
        string-keys are:

        - 'vmap' or 'v' for `jax.vmap`
        - 'pmap' or 'p' for `jax.pmap`
        - 'lax.map' or 'lax' for `jax.lax.map`

        In case `map` is passed as a function, it should produce a mapped
        function f_mapped of a general function f as: `f_mapped = map(f)`
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
        return map_forest_mean(ham_vg, map=map, in_axes=(0, ))(
            tuple(primals_samples.at(primals))
        )

    return mean_vg


def mean_hessp(ham: Callable, map="vmap", *args, **kwargs):
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
            in_axes=(0, None),
            map=map
        )(tuple(primals_samples.at(primals)), tangents)

    return mean_hp


def mean_metric(metric: Callable, map="vmap"):
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
            partial(metric, **primals_kw), in_axes=(0, None), map=map
        )(tuple(primals_samples.at(primals)), tangents)

    return mean_met
