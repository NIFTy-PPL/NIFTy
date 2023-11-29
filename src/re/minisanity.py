# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pprint
from typing import Any, NamedTuple, TypeVar

import jax
from jax import numpy as jnp

from .evi import Samples
from .tree_math import Vector, get_map

O = TypeVar('O')
I = TypeVar('I')


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
    map = get_map(map)
    if not isinstance(position_or_samples,
                      Samples) or len(position_or_samples) == 0:
        if isinstance(position_or_samples, Samples):
            assert len(position_or_samples) == 0
            position_or_samples = position_or_samples.pos
        samples = jax.tree_map(
            lambda x: x[jnp.newaxis, ...], position_or_samples
        )
    else:
        assert isinstance(position_or_samples, Samples)
        samples = position_or_samples.samples
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

    def make_pretty_string(x):
        rsq = x.reduced_chisq
        s = (
            f"reduced χ²:{rsq[0]:8.2}±{rsq[1]:8.2}"
            f", avg:{x.mean[0]:+9.2}±{x.mean[1]:8.2}"
            f", #dof:{int(x.ndof):7d}"
        )
        return s

    def is_leaf(l):
        return isinstance(l, ChiSqStats)

    ps = jax.tree_map(make_pretty_string, stat_tree, is_leaf=is_leaf)
    # HACK to make the most common primal types look pretty
    ps = ps.tree if isinstance(ps, Vector) else ps
    pp = pprint.PrettyPrinter()
    if isinstance(ps, dict):
        msg = ""
        for k in sorted(ps.keys()):
            v = ps[k]
            if isinstance(v, str):
                msg += f"{str(k):22s}:: {v}\n"
            else:
                msg += f"{str(k):22s}::\n{pp.pformat(v)}\n"
    elif not isinstance(ps, str):
        msg = pp.pformat(ps)
    else:
        msg = ps
    return stat_tree, msg
