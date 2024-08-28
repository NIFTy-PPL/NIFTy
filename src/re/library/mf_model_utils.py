# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig,

from typing import Union, Optional

from ..correlated_field import (
    RegularCartesianGrid,
    matern_amplitude,
    non_parametric_amplitude,
    WrappedCall)
from ..model import Model
from ..num.stats_distributions import lognormal_prior, normal_prior
from ..tree_math import Vector


def _add_prefix_to_keys(tree, prefix):
    """Recursively add a prefix to all keys in a nested dictionary."""
    if isinstance(tree, Vector):
        tree = tree.tree
    if isinstance(tree, dict):
        return {f"{prefix}_{key}": _add_prefix_to_keys(value, prefix) for key, value in
                tree.items()}
    else:
        return tree


def _remove_prefix_from_keys(tree, prefix):
    """Recursively remove a prefix from all keys in a nested dictionary."""
    if isinstance(tree, Vector):
        tree = tree.tree
    if isinstance(tree, dict):
        prefix_length = len(prefix) + 1  # includes the underscore
        return {key[prefix_length:]: _remove_prefix_from_keys(value, prefix) for key, value in
                tree.items() if key.startswith(prefix)}
    else:
        return tree


def _safe_set_default_or_call(arg: Union[callable, tuple, list, None],
                              default: callable):
    """Either sets the default distribution or the callable"""
    return _set_default_or_call(arg, default) if arg is not None else None


def _set_default_or_call(arg: Union[callable, tuple, list],
                         default: callable):
    """Either sets the default distribution or the callable"""
    if callable(arg):
        # TODO: do a check here that it is a valid distribution.
        return arg
    return default(*arg)


def _check_demands(model_name, kwargs, demands):
    """Check that all demands are provided in kwargs."""
    for key in demands:
        assert key in kwargs, (f'{key} not in {model_name}.\n'
                               f'Provide settings for {key}')


def _acquire_submodel(model: Optional[Union[Model, None]], prefix: str):
    """
    Acquire a submodel with prefixed domain keys.

    Parameters
    ----------
        model: Model, optional
            The original model.
        prefix: str
            The prefix to add to the domain keys.

    Returns
    -------
        model: Model
            A new Model instance with the prefixed domain
            keys and modified call method.
    """
    if model is None:
        return None

    if isinstance(model.domain, dict):
        new_domain = model.domain
    else:
        new_domain = model.domain.tree

    new_domain = _add_prefix_to_keys(new_domain, prefix)

    def call(x):
        return model(_remove_prefix_from_keys(x, prefix))

    return Model(call, domain=new_domain)


def _build_distribution_or_default(arg: Union[callable, tuple, list],
                                   key: str,
                                   default: callable,
                                   shape: tuple = (),
                                   dtype=None):
    """
    Build a distribution from an argument or use a default.

    Parameters
    ----------
    arg: Union[callable, tuple, list]
        The argument to be used for creating the distribution. If a callable
        is provided, it will be used directly. If a tuple or list is provided,
        the default callable will be used with the unpacked `arg` as its arguments.
    key: str
        The name or identifier for the distribution.
    default: callable
        The default callable to be used if `arg` is not a callable.
    shape: tuple, optional
        The shape of the resulting distribution. Defaults to an empty tuple `()`.
    dtype: type, optional
        The data type of the distribution. Defaults to None.

    Returns
    -------
    distribution: WrappedCall
        A WrappedCall instance representing the distribution with the specified
        `name`, `shape`, and `dtype`.
    """
    return WrappedCall(
        arg if callable(arg) else default(*arg),
        name=key,
        shape=shape,
        dtype=dtype,
        white_init=True)


def build_amplitude_model(
    grid: RegularCartesianGrid,
    settings: Optional[dict],
    amplitude_model: str = "non_parametric",
    renormalize_amplitude: bool = False,
    prefix: str = None,
    kind: str = "amplitude",
) -> Optional[Model]:
    """
    Build an amplitude model based on
    the specified settings and model type.

    Parameters
    ----------
    grid: RegularCartesianGrid
        The grid on which the amplitude model is defined.
    settings: dict, optional
        A dictionary of settings that configure
        the amplitude model.
        If None, the builder returns None.
    amplitude_model: str, optional
        The type of amplitude model to build.
        Must be either "non_parametric" or "matern".
        Defaults to "non_parametric".
    renormalize_amplitude: bool, optional
        Whether to renormalize the amplitude in the "matern" model.
        Defaults to False.
    prefix: str, optional
        A prefix to add to the domain keys.
        Defaults to None.
    kind: str, optional
        A string to specify the kind of amplitude.
        Defaults to "amplitude".

    Returns
    -------
    model: Model or None
        A Model instance representing the amplitude model
        configured with the provided settings.
        Returns None if `settings` is None.

    Raises
    ------
    ValueError
        If `amplitude_model` is not "non_parametric" or "matern".
    """

    if settings is None:
        return None

    key = f'{prefix}_amplitude_' if prefix is not None else 'amplitude_'

    if amplitude_model == "non_parametric":
        _check_demands(key, settings, demands={'fluctuations', 'loglogavgslope',
                                               'flexibility', 'asperity'})
        return non_parametric_amplitude(
            grid,
            fluctuations=_set_default_or_call(settings['fluctuations'],
                                              lognormal_prior),
            loglogavgslope=_set_default_or_call(settings['loglogavgslope'],
                                                normal_prior),
            flexibility=_safe_set_default_or_call(settings['flexibility'],
                                                  lognormal_prior),
            asperity=_safe_set_default_or_call(settings['asperity'],
                                               lognormal_prior),
            prefix=key,
        )
    elif amplitude_model == "matern":
        _check_demands(
            key, settings, demands={'scale', 'cutoff', 'loglogslope'})
        return matern_amplitude(
            grid,
            scale=_set_default_or_call(settings['scale'],
                                       lognormal_prior),
            cutoff=_set_default_or_call(settings['cutoff'],
                                        lognormal_prior),
            loglogslope=_set_default_or_call(settings['loglogslope'],
                                             normal_prior),
            renormalize_amplitude=renormalize_amplitude,
            kind=kind,
            prefix=key)
    else:
        raise ValueError("Type must be 'non_parametric' or 'matern'.")
