# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from typing import Any, Callable, Dict, Hashable, Mapping, TypeVar, Union

import jax
from jax import numpy as jnp, tree_map

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


def ducktape(call: Callable[[I], O],
             key: Hashable) -> Callable[[Mapping[Hashable, I]], O]:
    def named_call(p):
        return call(p[key])

    return named_call


def ducktape_left(call: Callable[[I], O],
                  key: Hashable) -> Callable[[I], Dict[Hashable, O]]:
    def named_call(p):
        return {key: call(p)}

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

def _red_chisq(inp):
    #FIXME complex numbers (use nifty convention)
    return jnp.vdot(inp.conjugate(), inp).real / inp.size

def reduced_chisq_stats(primals, samples = None, func = None):
    """Computes the reduced chi-squared summary statistics for given input.

    Parameters:
    -----------
    primals: tree-like
        Input values to compute reduces chi-sq statistics. The statistics is 
        computed for each leaf of the pytree, i.E. only array-like leafs are
        square averaged. See `samples` and `func` for further infos.
    samples: Samples (optional)
        Posterior samples corresponding to primals. If provided, the chi-sq
        statistics is computed for each sample, and the sample 
        mean and standard deviation of the statistics is returned.
    func: Callable (optional)
        Function to compute the chi-sq statistics for instead of primals 
        (samples). If provided, the statistics is computed for `func(x)` instead
        of `x` where x is either primals or a sample.

    Returns:
    --------
    reduced_chisq: tree-like
        Pytree of Mean-Std pairs of the reduces chi-sq statistics at each leaf 
        of the tree. Irregardless of samples being provided or not, the 
        resulting leafs are always Mean-Std pairs, with the Std always being
        zero if samples is None.
    """
    if samples is not None:
        samples = samples.at(primals).samples
    else:
        samples = jax.tree_map(lambda x: x[jnp.newaxis, ...], primals)
    samples = jax.vmap(func)(samples) if func is not None else samples

    def red_chisq_stat(s):
        res = jax.vmap(_red_chisq)(s)
        return (jnp.mean(res), jnp.std(res))

    return jax.tree_map(red_chisq_stat, samples)