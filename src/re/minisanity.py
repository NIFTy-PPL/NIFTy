# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pprint
from typing import Any, NamedTuple, TypeVar

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map

from .evi import Samples
from .tree_math import Vector, get_map

O = TypeVar("O")
I = TypeVar("I")


def _residual_params(inp):
    ndof = inp.size if jnp.isrealobj(inp) else 2 * inp.size
    mean = jnp.sum(inp) / inp.size
    rchisq = jnp.vdot(inp, inp).real / ndof
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
    if not isinstance(position_or_samples, Samples) or len(position_or_samples) == 0:
        if isinstance(position_or_samples, Samples):
            assert len(position_or_samples) == 0
            position_or_samples = position_or_samples.pos
        samples = tree_map(lambda x: x[jnp.newaxis, ...], position_or_samples)
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

    return tree_map(red_chisq_stat, samples)


def _rpprint(ps: Any, *, _indent=0, _key="") -> str:
    if isinstance(ps, Vector):
        msg = _rpprint(ps.tree, _indent=_indent)
    elif isinstance(ps, dict):
        msg = ""
        for k, v in ps.items():
            k = _key + "/" * bool(_key) + str(k)
            if isinstance(v, dict):
                msg += _rpprint(v, _indent=_indent, _key=k)
            else:
                msg += "  " * _indent + f"{k:24s}::"
                m = _rpprint(v, _indent=_indent + 1, _key="")
                msg += (" " + m.lstrip()) if len(m.splitlines()) == 1 else ("\n" + m)
    elif isinstance(ps, (tuple, list)):
        msg = f"{'list' if isinstance(ps, list) else 'tuple'}(\n"
        for v in ps:
            msg += _rpprint(v, _indent=_indent + 1, _key="")
        msg += ")\n"
    else:
        # Catch all other instances with PrettyPrinter
        msg = "  " * _indent + pprint.pformat(ps) + "\n"
    return msg


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

    ps = tree_map(make_pretty_string, stat_tree, is_leaf=is_leaf)
    return stat_tree, _rpprint(ps)
