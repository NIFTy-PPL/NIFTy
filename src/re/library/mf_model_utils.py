# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig,

from typing import Union, Optional

from ..correlated_field import (
    RegularCartesianGrid,
    MaternAmplitude,
    NonParametricAmplitude,
    WrappedCall)
from ..num.stats_distributions import lognormal_prior, normal_prior


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


def build_normalized_amplitude_model(
    grid: RegularCartesianGrid,
    settings: Optional[dict],
    amplitude_model: str = "non_parametric",
    renormalize_amplitude: bool = True,
    prefix: str = None,
    kind: str = "amplitude",
) -> Union[None, MaternAmplitude, NonParametricAmplitude]:
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
        Defaults to True.
    prefix: str, optional
        A prefix to add to the domain keys.
        Defaults to None.
    kind: str, optional
        A string to specify the kind of amplitude.
        Defaults to "amplitude".

    Returns
    -------
    amplitude: Amplitude or None
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
        _check_demands(
            key,
            settings,
            demands={'loglogavgslope', 'flexibility', 'asperity'}
        )
        return NonParametricAmplitude(
            grid,
            fluctuations=None,
            loglogavgslope=_set_default_or_call(
                settings['loglogavgslope'], normal_prior),
            flexibility=_safe_set_default_or_call(
                settings['flexibility'], lognormal_prior),
            asperity=_safe_set_default_or_call(
                settings['asperity'], lognormal_prior),
            prefix=key,
        )
    elif amplitude_model == "matern":
        _check_demands(
            key, settings, demands={'cutoff', 'loglogslope'})
        return MaternAmplitude(
            grid,
            scale=None,
            cutoff=_set_default_or_call(
                settings['cutoff'], lognormal_prior),
            loglogslope=_set_default_or_call(
                settings['loglogslope'], normal_prior),
            renormalize_amplitude=renormalize_amplitude,
            kind=kind,
            prefix=key)
    else:
        raise ValueError("Type must be 'non_parametric' or 'matern'.")
