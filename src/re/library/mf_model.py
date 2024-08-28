# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig,
# Vincent Eberle, Philipp Frank, Vishal Johnson,
# Jakob Roth, Margret Westerkamp

from functools import reduce, partial
from typing import Optional, Union, Callable

import jax.numpy as jnp
from jax import vmap
from numpy.typing import ArrayLike

from .frequency_deviations import build_frequency_deviations_model
from .mf_model_utils import (_acquire_submodel,
                             _build_distribution_or_default,
                             build_amplitude_model)
from .. import ShapeWithDtype
from ..correlated_field import (
    _make_grid,
    hartley,
    RegularCartesianGrid)
from ..model import Model
from ..num.stats_distributions import lognormal_prior, normal_prior


class CorrelatedMultiFrequencySky(Model):
    """
    A model for generating a correlated multi-frequency sky map based on
    spatial and spectral correlation models.
    """
    def __init__(
            self,
            prefix: str,
            grid: RegularCartesianGrid,
            relative_log_frequencies: Union[tuple[float], ArrayLike],
            zero_mode: Model,
            zero_mode_offset: float,
            spatial_amplitude: Model,
            spectral_index_mean: Model,
            spectral_index_fluctuations: Model,
            spectral_amplitude: Optional[Model] = None,
            spectral_index_deviations: Optional[Model] = None,
            nonlinearity: Optional[Callable] = jnp.exp,
            dtype: type = jnp.float64,
    ):
        self.prefix = prefix
        slicing_tuple = (slice(None),) + (None,) * len(grid.shape)
        self._freqs = jnp.array(relative_log_frequencies)[slicing_tuple]
        self.hdvol = 1.0 / grid.total_volume
        self.pd = grid.harmonic_grid.power_distributor
        self.ht = partial(hartley, axes=tuple(range(len(grid.shape))))
        self._nonlinearity = nonlinearity
        self.zero_mode_offset = zero_mode_offset
        self._zm = _acquire_submodel(
            zero_mode, prefix)
        self.spatial_amplitude = _acquire_submodel(
            spatial_amplitude, prefix)
        self.spectral_index_mean = _acquire_submodel(
            spectral_index_mean, prefix)
        self.spectral_index_fluctuations = _acquire_submodel(
            spectral_index_fluctuations, prefix)
        self.spectral_index_deviations_model = _acquire_submodel(
            spectral_index_deviations, prefix)
        self.spectral_amplitude = _acquire_submodel(spectral_amplitude,
                                                    prefix)

        models = [self._zm,
                  self.spatial_amplitude,
                  self.spectral_index_mean,
                  self.spectral_index_fluctuations,
                  self.spectral_index_deviations_model,
                  self.spectral_amplitude]

        domain = reduce(
            lambda a, b: a | b, [m.domain for m in models if m is not None]
        )

        domain[f"{self.prefix}_spatial_xi"] = ShapeWithDtype(
            grid.shape, dtype)
        domain[f"{self.prefix}_spectral_index_xi"] = ShapeWithDtype(
            grid.shape, dtype)

        super().__init__(self._build_apply(), domain=domain)

    def _build_apply(self):
        spectral_amplitude_flag = self.spectral_amplitude is not None
        if self.spectral_index_deviations_model is not None:

            def apply_with_deviations(p):
                """
                Apply method of the model with spectral
                index deviations.
                Implements:
                .. math::
                        sky = \\nonlinearity(F[A_spatial *
                        io(k, \\nu_0) + A_spectral *
                        (slope(k) * (\\nu-\\nu_0) +
                        GaussMarkovProcess(k, \\nu-\\nu_0)
                        - AvgSlope[GaussMarkovProcess]
                        )] + zero_mode)
                where :math:`F` is the Fourier transform,
                :math:`k` is the spatial frequency index,
                :math:`\\nu` is the spectral frequency index,
                and `slope` represents the spectral index.
                """
                zm = self.zero_mode(p)

                amplitude = self.spatial_amplitude(p)
                amplitude = amplitude.at[0].set(0.0)
                spatial_xi = p[f"{self.prefix}_spatial_xi"]
                spec_idx_xis = p[f"{self.prefix}_spectral_index_xi"]

                spectral_index = self.spectral_index_fluctuations(
                    p) * spec_idx_xis
                spec_idx_mean = self.spectral_index_mean(p)
                deviations = self.spectral_index_deviations_model(p)
                distributed_amplitude = amplitude[self.pd]

                if spectral_amplitude_flag:
                    spectral_amplitude = self.spectral_amplitude(p)
                    spectral_amplitude = spectral_amplitude.at[0].set(0.0)
                    distributed_spectral_amplitude = spectral_amplitude[self.pd]
                    spectral_terms = spectral_index * self._freqs + deviations
                    ht_values = vmap(self.ht)(
                        distributed_amplitude * spatial_xi +
                        distributed_spectral_amplitude * spectral_terms)
                else:
                    terms = spectral_index*self._freqs + spatial_xi + deviations
                    ht_values = vmap(self.ht)(distributed_amplitude * terms)
                return self._nonlinearity(self.hdvol * ht_values +
                                          spec_idx_mean * self._freqs + zm)

            return apply_with_deviations

        def apply_without_deviations(p):
            """
            Apply method of the model without spectral
            index deviations.
            Implements:
            .. math::
                    sky = \\nonlinearity(F[A_spatial *
                    io(k, \\nu_0) + A_spectral *
                    (slope(k) * (\\nu-\\nu_0))]
                    + zero_mode)
            where :math:`F` is the Fourier transform,
            :math:`k` is the spatial frequency index,
            :math:`\\nu` is the spectral frequency index,
            and `slope` represents the spectral index.
            """
            zm = self.zero_mode(p)
            amplitude = self.spatial_amplitude(p)
            amplitude = amplitude.at[0].set(0.0)

            spatial_xi = p[f"{self.prefix}_spatial_xi"]
            spec_idx_xis = p[f"{self.prefix}_spectral_index_xi"]
            spectral_index = self.spectral_index_fluctuations(p) * spec_idx_xis
            spec_idx_mean = self.spectral_index_mean(p)
            spatial_offset = self.hdvol*self.ht(amplitude[self.pd]*spatial_xi)

            if spectral_amplitude_flag:
                amplitude = self.spectral_amplitude(p)
                amplitude = amplitude.at[0].set(0.0)
            spectral_index_spatial = (self.hdvol*self.ht(amplitude[self.pd]*spectral_index)
                                      + spec_idx_mean)
            return self._nonlinearity(spatial_offset + zm
                                      + spectral_index_spatial * self._freqs)

        return apply_without_deviations

    def spatial_distribution(self, p):
        """Convenience function to retrieve the model's spatial distribution."""
        amplitude = self.spatial_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        spatial_xi = p[f"{self.prefix}_spatial_xi"]
        return self.hdvol*self.ht(amplitude[self.pd]*spatial_xi)

    def spectral_index(self, p):
        """Convenience function to retrieve the model's spectral index."""
        amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        spec_idx_fluctuations = self.spectral_index_fluctuations(p)
        spec_idx_xis = p[f"{self.prefix}_spectral_index_xi"]
        return (self.hdvol*self.ht(amplitude[self.pd]*spec_idx_fluctuations*spec_idx_xis)
                + self.spectral_index_mean(p))

    def spectral_deviations(self, p):
        """Convenience function to retrieve the model's spectral deviations."""
        if self.spectral_index_deviations_model is None:
            return None
        else:
            amplitude = self.spectral_amplitude(p)
            amplitude = amplitude.at[0].set(0.0)
            return self.hdvol * vmap(self.ht)(amplitude[self.pd] *
                                      self.spectral_index_deviations_model(p))

    def zero_mode(self, p):
        """Convenience function to retrieve the model's zero mode."""
        return self._zm(p) + self.zero_mode_offset


def build_default_mf_model(
    prefix: str,
    shape: tuple[int],
    distances: tuple[float],
    log_frequencies: Union[tuple[float], ArrayLike],
    reference_frequency_index: int,
    zero_mode_settings: dict,
    spatial_amplitude_settings: dict,
    spectral_index_settings: dict,
    spectral_amplitude_settings: Optional[dict] = None,
    deviations_settings: Optional[dict] = None,
    spatial_amplitude_model: str = "non_parametric",
    spectral_amplitude_model: str = "non_parametric",
    harmonic_type: str = 'fourier',
) -> CorrelatedMultiFrequencySky:
    """
    Builds a multi-frequency sky model parametrized as

    .. math ::
        sky = \\exp(F[A_spatial *
              io(k, \\nu_0) + A_spectral *
              (slope(k) * (\\nu-\\nu_0) +
              GaussMarkovProcess(k, \\nu-\\nu_0)
              - AvgSlope[GaussMarkovProcess]
              )] + zero_mode)


    Parameters
    ----------
    prefix: str
        The prefix of the multi-frequency model.

    shape: tuple
        The shape of the spatial_amplitude domain.

    distances: tuple
        The distances of the spatial_amplitude domain.

    log_frequencies: tuple, list, ArrayLike
        Array of logarithmically spaced frequencies.

    reference_frequency_index: int
        Index of the reference frequency in `log_frequencies`.

    zero_mode_settings: dict
        Settings for the zero mode priors.
        Should contain the following keys:
            - mean: zero mode mean
            - deviations: callable or parameters
            for default (normal prior)

    spatial_amplitude_settings: dict
        Settings for the amplitude model priors.
        Should contain the following keys:
            For correlated field amplitude:
            - fluctuations: callable or parameters
                     for default (lognormal prior)
            - loglogavgslope: callable or parameters
                     for default (lognormal prior)
            - flexibility: callable or parameters or None
                     for default (lognormal prior)
            - asperity: callable or parameters or None
                     for default (lognormal prior)

            For Matérn amplitude:
            - scale: callable or parameters
                     for default (lognormal prior)
            - cutoff: callable or parameters
                     for default (lognormal prior)
            - loglogslope: callable or parameters
                     for default (lognormal prior)

    spectral_index_settings: dict
        Settings for the spectral index priors.
        Should contain the following keys:
            - mean: callable or parameters
                 for default (normal prior)
            - fluctuations: callable or parameters
                for default (lognormal prior)

    spectral_amplitude_settings: dict, opt
        If `None` the spectral amplitude is the same as the spatial amplitude.
        If not `None` sets the spectral amplitude settings.
        Should be formatted as `spatial_amplitude_settings`.

    deviations_settings: dict, opt
        Settings for the spectral index priors.
        If none deviations are not build.
        Should contain the following keys:
        - process: wiener (default)
        - sigma: callable or parameters
             for default (lognormal prior)

    spatial_amplitude_model: str
        Type of the spatial amplitude model to be used.
        By default, the correlated field model
        (`'non_parametric'`).

    spectral_amplitude_model: str
        Type of the spectral amplitude model to be used.
        By default, the correlated field model
        (`'non_parametric'`).

    harmonic_type: str
        The type of the harmonic domain for the amplitude model.

    Returns
    -------
    model: CorrelatedMultiFrequencySky
        The multi-frequency sky model
    """

    grid = _make_grid(shape, distances, harmonic_type)

    spatial_model = build_amplitude_model(
        grid,
        spatial_amplitude_settings,
        amplitude_model=spatial_amplitude_model)

    spectral_model = build_amplitude_model(
        grid,
        spectral_amplitude_settings,
        amplitude_model=spectral_amplitude_model
        )

    deviations_model = build_frequency_deviations_model(shape,
                                                        log_frequencies,
                                                        reference_frequency_index,
                                                        deviations_settings)

    zero_mode = _build_distribution_or_default(
        zero_mode_settings['deviations'],
        f'zero_mode',
        normal_prior
    )
    spectral_index_mean = _build_distribution_or_default(
        spectral_index_settings['mean'],
        f'spectral_index_mean',
        normal_prior
    )
    spectral_index_fluctuations = _build_distribution_or_default(
        spectral_index_settings['fluctuations'],
        f'spectral_index_fluctuations',
        lognormal_prior
    )

    return CorrelatedMultiFrequencySky(
        prefix=prefix,
        grid=grid,
        relative_log_frequencies=jnp.array(
            log_frequencies) - log_frequencies[reference_frequency_index],
        zero_mode=zero_mode,
        zero_mode_offset=zero_mode_settings['mean'],
        spatial_amplitude=spatial_model,
        spectral_index_mean=spectral_index_mean,
        spectral_amplitude=spectral_model,
        spectral_index_fluctuations=spectral_index_fluctuations,
        spectral_index_deviations=deviations_model
    )
