# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pprint
from typing import Any, Callable, Dict, Hashable, Mapping, NamedTuple, TypeVar

import jax
from jax import numpy as jnp

O = TypeVar('O')
I = TypeVar('I')


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def split(mappable, keys):
    """Split a dictionary into one containing only the specified keys and one
    with all of the remaining ones.
    """
    sel, rest = {}, {}
    for k, v in mappable.items():
        if k in keys:
            sel[k] = v
        else:
            rest[k] = v
    return sel, rest


def isiterable(candidate):
    try:
        iter(candidate)
        return True
    except (TypeError, AttributeError):
        return False


def is1d(ls: Any) -> bool:
    """Indicates whether the input is one dimensional.

    An object is considered one dimensional if it is an iterable of
    non-iterable items.
    """
    if hasattr(ls, "ndim"):
        return ls.ndim == 1
    if not isiterable(ls):
        return False
    return all(not isiterable(e) for e in ls)


def doc_from(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def wrap(
    call: Callable[[I], O],
    name: Hashable,
) -> Callable[[Mapping[Hashable, I]], O]:
    def named_call(p):
        return call(p[name])

    return named_call


def wrap_left(
    call: Callable[[I], O],
    name: Hashable,
) -> Callable[[I], Dict[Hashable, O]]:
    def named_call(p):
        return {name: call(p)}

    return named_call


def interpolate(xmin=-7., xmax=7., N=14000) -> Callable:
    """Replaces a local nonlinearity such as jnp.exp with a linear interpolation

    Interpolating functions speeds up code and increases numerical stability in
    some cases, but at a cost of precision and range.

    Parameters
    ----------
    xmin : float
        Minimal interpolation value. Default: -7.
    xmax : float
        Maximal interpolation value. Default: 7.
    N : int
        Number of points used for the interpolation. Default: 14000
    """
    def decorator(f):
        from functools import wraps

        x = jnp.linspace(xmin, xmax, N)
        y = f(x)

        @wraps(f)
        def wrapper(t):
            return jnp.interp(t, x, y)

        return wrapper

    return decorator


def _residual_params(inp):
    ndof = inp.size if jnp.isrealobj(inp) else 2 * inp.size
    mean = jnp.sum(inp.real + inp.imag) / ndof
    rchisq = jnp.vdot(inp, inp) / ndof
    return mean, rchisq, ndof


class ChiSqStats(NamedTuple):
    mean: Any
    reduced_chisq: Any
    ndof: Any


def reduced_residual_stats(position_or_samples, func=None, *, map="lmap"):
    """Computes the average, reduced chi-squared, and number of parameters
    as a summary statistics for a given input.

    Parameters:
    -----------
    position_or_samples: tree-like or Samples
        Input values to compute reduces chi-sq statistics. The statistics is
        computed for each leaf of the pytree, i.E. only array-like leafs are
        square averaged. If `positin_or_samples` is a `Sample` object, the
        chi-sq statistics is computed for each sample, and the sample mean and
        standard deviation of the statistics is returned.
    func: Callable (optional)
        Function to apply to `position_or_samples` before computing the chi-sq
        statistics for. If provided, the statistics is computed for `func(x)`
        instead of `x` where x is either primals or a sample.

    Returns:
    --------
    stats: tree-like
        Pytree of tuple containing the mean, reduced chi-squared, and number of
        parameters for each leaf of the input tree. For the mean and reduched
        chi-sq, a numpy array with the sample mean and sample std is returned.
        If samples is None, the second entry of this array is always zero.
    """
    from .tree_math import get_map
    from .evi import Samples

    map = get_map(map)
    if isinstance(position_or_samples, Samples):
        samples = position_or_samples.samples
    else:
        samples = jax.tree_map(
            lambda x: x[jnp.newaxis, ...], position_or_samples
        )
    samples = map(func)(samples) if func is not None else samples

    get_stats = map(_residual_params)

    def red_chisq_stat(s):
        m, rx, nd = get_stats(s)
        m = jnp.array([jnp.mean(m), jnp.std(m)])
        rx = jnp.array([jnp.mean(rx), jnp.std(rx)])
        return ChiSqStats(m, rx, nd[0])

    return jax.tree_map(red_chisq_stat, samples)


def minisanity(position_or_samples, func=None, *, map="lmap"):
    """Wrapper for `reduced_residual_stats` to retrieve the reduced chi-squared
    and a pretty-printable string of the statistics."""
    stat_tree = reduced_residual_stats(position_or_samples, func=func, map=map)

    def pretty_string(x):
        rsq = x.reduced_chisq
        s = (
            f"reduced χ²: {rsq[0]:.2}±{rsq[1]:.2}"
            f", avg: {x.mean[0]:.2}±{x.mean[1]:.2}"
            f", #dof: {int(x.ndof)}"
        )
        return s

    def is_leaf(l):
        return isinstance(l, ChiSqStats)

    stat_tree = jax.tree_map(pretty_string, stat_tree, is_leaf=is_leaf)
    return stat_tree, pprint.PrettyPrinter().pformat(stat_tree)
