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
from ..gauss_markov import GaussMarkovProcess, build_wiener_process
from ..model import Model


class FrequencyDeviations(Model):
    """
    Model for calculating frequency deviations relative to a reference frequency.

    Parameters
    ----------
    frequency_deviations_model: GaussMarkovProcess
        A Gauss-Markov process used to model frequency deviations.
        TODO: Extend to allow for correlated field.
    frequencies: Union[tuple[float], ArrayLike]
        A list or array of frequencies at which deviations are calculated.
    reference_frequency_index: int
        The index of the reference frequency within the `frequencies` array.
    """
    def __init__(
        self,
        frequency_deviations_model: GaussMarkovProcess,  # TODO: Allow for correlated field
        frequencies: Union[tuple[float], ArrayLike],
        reference_frequency_index: int
    ):
        self.frequency_deviations = frequency_deviations_model

        slicing_tuple = (
            (slice(None),) +
            (None,) * len(self.frequency_deviations.target.shape[1:])
        )
        relative_freqs = jnp.array(
            frequencies - frequencies[reference_frequency_index])[slicing_tuple]
        frequencies_denominator = 1 / jnp.sum(relative_freqs**2)

        def deviations_call(p):
            dev = self.frequency_deviations(p)
            dev = dev - dev[reference_frequency_index]

            # m = sum_l(gmp(l) * (l-l0)) / sum_l((l-l0)**2)
            dev_slope = (jnp.sum(dev*relative_freqs, axis=0)
                         * frequencies_denominator)

            return dev - dev_slope * relative_freqs

        super().__init__(
            deviations_call, init=frequency_deviations_model.init)


def build_frequency_deviations_model(
    shape: tuple[int],
    log_frequencies: Union[tuple[float], ArrayLike],
    reference_frequency_index: int,
    deviations_settings: Optional[dict],
    prefix: str = None,
) -> FrequencyDeviations | None:
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
    FrequencyDeviations or None
        A FrequencyDeviations instance configured with
        the specified settings.
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

        domain_key = f'{dev_name}_wp'
        process = build_wiener_process(
            x0=jnp.zeros(shape),  # sets x0 to 0 to avoid degeneracies
            sigma=sigma,
            dt=log_frequencies[1:]-log_frequencies[:-1],
            name=domain_key,
        )
    else:
        raise NotImplementedError(f'{process_name} not implemented.')

    return FrequencyDeviations(process, log_frequencies, reference_frequency_index)
