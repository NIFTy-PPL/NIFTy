# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig
# Vincent Eberle, Philipp Frank, Vishal Johnson,
# Jakob Roth, Margret Westerkamp

from typing import Union, Optional

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .mf_model_utils import _check_demands, _build_distribution_or_default
from .. import lognormal_prior
from ..gauss_markov import GaussMarkovProcess, build_fixed_point_wiener_process


def build_frequency_deviations_model_with_degeneracies(
    shape: tuple[int],
    log_frequencies: Union[tuple[float], ArrayLike],
    reference_frequency_index: int,
    deviations_settings: Optional[dict],
    prefix: str = None,
) -> GaussMarkovProcess | None:
    """
    Builds a frequency deviations model based on
    the specified settings.

    Parameters
    ----------
    shape: tuple[int]
        The shape of the target grid for the deviations
        model.
    log_frequencies: Union[tuple[float], ArrayLike]
        A list or array of logarithmic frequencies.
    reference_frequency_index: int
        The index of the reference frequency within
        the `log_frequencies` array.
    deviations_settings: dict, optional
        A dictionary of settings that configure
        the frequency deviations model.
        If None, the function returns None.
    prefix: str, optional
        A prefix to add to the domain keys.
        Defaults to None.

    Returns
    -------
    GaussMarkovProcess or None
        A GaussMarkovProcess instance configured with the specified settings.
        Returns None if `deviations_settings` is None.

    Raises
    ------
    NotImplementedError
        If the specified process in `deviations_settings`
        is not implemented.
    """
    if deviations_settings is None:
        return None

    if isinstance(log_frequencies, tuple):
        log_frequencies = np.array(log_frequencies)

    dev_name = f'{prefix}_deviations' if prefix is not None else 'deviations'
    process_name = deviations_settings.get('process', 'wiener').lower()

    # FIXME: Add more processes when they are vectorized
    if process_name in {'wiener', 'wiener_process'}:
        _check_demands(dev_name, deviations_settings, demands={'sigma'})
        sigma = _build_distribution_or_default(deviations_settings.get('sigma'),
                                               f"{dev_name}_sigma",
                                               lognormal_prior)

        domain_key = f'{dev_name}_xi'
        process = build_fixed_point_wiener_process(
            x0=jnp.zeros(shape),  # sets x0 to 0 to avoid degeneracies
            sigma=sigma,
            t=log_frequencies,
            reference_t_index=reference_frequency_index,
            name=domain_key,
        )
    else:
        raise NotImplementedError(f'{process_name} not implemented.')

    return process
