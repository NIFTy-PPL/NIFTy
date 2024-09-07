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
from .mf_model_utils import (_build_distribution_or_default,
                             _check_demands,
                             build_amplitude_model)
from .. import ShapeWithDtype, logger
from ..correlated_field import (
    RegularCartesianGrid,
    _make_grid,
    hartley,
    Amplitude)
from ..model import Model
from ..num.stats_distributions import lognormal_prior, normal_prior


def _amplitude_without_fluctuations(amplitude: Amplitude) -> Amplitude:
    """Return an amplitude model without fluctuations."""
    def spectral_amplitude_func(x): return amplitude(
        x) / amplitude.fluctuations(x)

    return Amplitude(
        spectral_amplitude_func,
        amplitude.domain,
        amplitude.grid,
        lambda _: 1.,
    )


class CorrelatedMultiFrequencySky(Model):
    """
    A model for generating a correlated multi-frequency sky map based on
    spatial and spectral correlation models.
    """

    def __init__(
        self,
        prefix: str,
        relative_log_frequencies: Union[tuple[float], ArrayLike],
        zero_mode: Model,
        spatial_fluctuations: Model,
        spatial_amplitude: Amplitude, # TODO: make this a normalized amplitude model
        spectral_index_mean: Model,
        spectral_index_fluctuations: Model,
        spectral_amplitude: Optional[Amplitude] = None, # TODO: make this a normalized amplitude model
        spectral_index_deviations: Optional[Model] = None,
        nonlinearity: Optional[Callable] = jnp.exp,
        dtype: type = jnp.float64,
    ):
        if not isinstance(spatial_amplitude, Amplitude):
            raise ValueError("`spatial_amplitude` must be an amplitude model.")

        grid = spatial_amplitude.grid
        slicing_tuple = (slice(None),) + (None,) * len(grid.shape)
        self.prefix = prefix
        self._freqs = jnp.array(relative_log_frequencies)[slicing_tuple]
        self.hdvol = 1.0 / grid.total_volume
        self.pd = grid.harmonic_grid.power_distributor
        self.ht = partial(hartley, axes=tuple(range(len(grid.shape))))
        self._nonlinearity = nonlinearity

        self.zero_mode = zero_mode
        self.spatial_fluctuations = spatial_fluctuations
        self.spatial_amplitude = _amplitude_without_fluctuations(
            spatial_amplitude) # TODO: make this a normalized amplitude model
        self.spectral_index_mean = spectral_index_mean
        self.spectral_index_fluctuations = spectral_index_fluctuations
        self.spectral_index_deviations = spectral_index_deviations

        if spectral_amplitude is None: # TODO: make this a normalized amplitude model
            self.spectral_amplitude = None
        else:
            self.spectral_amplitude = _amplitude_without_fluctuations(
                spectral_amplitude)

        models = [
            self.zero_mode,
            self.spatial_fluctuations,
            self.spatial_amplitude,
            self.spectral_index_mean,
            self.spectral_index_fluctuations,
            self.spectral_index_deviations,
            self.spectral_amplitude,
        ]

        domain = reduce(
            lambda a, b: a | b, [m.domain for m in models if m is not None]
        )

        domain[f"{self.prefix}_spatial_xi"] = ShapeWithDtype(
            grid.shape, dtype)
        domain[f"{self.prefix}_spectral_index_xi"] = ShapeWithDtype(
            grid.shape, dtype)

        super().__init__(domain=domain)

    def __call__(self, p):
        if self.spectral_index_deviations is not None:
            """
            Apply method of the model with spectral
            index deviations.
            Implements:
            .. math::
                    sky = \\nonlinearity(
                    F[A_spatial*io(k, \\mu_0) + 
                      A_spectral*(
                        slope_fluctuations(k)*(\\mu-\\mu_0)
                        + GaussMarkovProcess(k, \\mu-\\mu_0)
                        - AvgSlope[GaussMarkovProcess]*(\\mu-\\mu_0)
                      )] + slope_mean*(\\mu-\\mu_0) + zero_mode)
            where :math:`F` is the Fourier transform,
            :math:`k` is the spatial frequency index,
            :math:`\\mu` is the log spectral frequency index,
            and `slope` represents the spectral index.
            """
            zm = self.zero_mode(p)

            spat_amplitude = self.spatial_amplitude(p)
            spat_amplitude = spat_amplitude.at[0].set(0.0)
            distributed_spatial_amplitude = spat_amplitude[self.pd]

            spatial_xis = p[f"{self.prefix}_spatial_xi"]
            spectral_xis = p[f"{self.prefix}_spectral_index_xi"]
            spatial_reference = self.spatial_fluctuations(p) * spatial_xis
            spectral_index_mean = self.spectral_index_mean(p)
            spectral_terms = (self.spectral_index_fluctuations(p) *
                              spectral_xis * self._freqs +
                              self.spectral_index_deviations(p))

            if self.spectral_amplitude is None:
                cf_values = self.hdvol*vmap(self.ht)(
                    distributed_spatial_amplitude *
                    (spatial_reference + spectral_terms)
                )
            else:
                spec_amplitude = self.spectral_amplitude(p)
                spec_amplitude = spec_amplitude.at[0].set(0.0)
                distributed_spectral_amplitude = spec_amplitude[self.pd]

                cf_values = self.hdvol*vmap(self.ht)(
                    distributed_spatial_amplitude * spatial_reference +
                    distributed_spectral_amplitude * spectral_terms)

            return self._nonlinearity(
                cf_values + spectral_index_mean*self._freqs + zm)

        else:
            """
            Apply method of the model without spectral
            index deviations.
            Implements:
            .. math::
                    sky = \\nonlinearity(
                    F[A_spatial * io(k, \\mu_0)] +
                    (F[A_spectral * slope_fluctuations(k)] + slope_mean)*(\\mu-\\mu_0)
                    + zero_mode)
            where :math:`F` is the Fourier transform,
            :math:`k` is the spatial frequency index,
            :math:`\\mu` is the log spectral frequency index,
            and `slope` represents the spectral index.
            """
            zm = self.zero_mode(p)
            spat_amplitude = self.spatial_amplitude(p)
            spat_amplitude = spat_amplitude.at[0].set(0.0)

            spatial_xis = p[f"{self.prefix}_spatial_xi"]
            spectral_xis = p[f"{self.prefix}_spectral_index_xi"]
            spatial_reference = self.spatial_fluctuations(p) * spatial_xis
            spectral_index_mean = self.spectral_index_mean(p)

            if self.spectral_amplitude is None:
                spec_amplitude = spat_amplitude
            else:
                spec_amplitude = self.spectral_amplitude(p)
                spec_amplitude = spec_amplitude.at[0].set(0.0)

            correlated_spatial_reference = self.hdvol*self.ht(
                spat_amplitude[self.pd]*spatial_reference)
            correlated_spectral_index = self.hdvol * self.ht(
                spec_amplitude[self.pd] *
                self.spectral_index_fluctuations(p) *
                spectral_xis)

            return self._nonlinearity(
                correlated_spatial_reference +
                (correlated_spectral_index + spectral_index_mean)*self._freqs +
                zm)

    def reference_frequency_distribution(self, p):
        """Convenience function to retrieve the model's spatial distribution at
        the reference frequency."""
        amplitude = self.spatial_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        spatial_xi = p[f"{self.prefix}_spatial_xi"]
        return self._nonlinearity(
            self.hdvol*self.ht(
                amplitude[self.pd] * self.spatial_fluctuations(p) * spatial_xi)
            + self.zero_mode(p)
        )

    def spectral_index_distribution(self, p):
        """Convenience function to retrieve the model's spectral index."""
        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        spectral_xi = p[f"{self.prefix}_spectral_index_xi"]
        amplitude = amplitude.at[0].set(0.0)
        spectral_index_mean = self.spectral_index_mean(p)

        return (self.hdvol*self.ht(amplitude[self.pd] *
                                   self.spectral_index_fluctuations(p) *
                                   spectral_xi)
                + spectral_index_mean)

    def spectral_deviations_distribution(self, p):
        """Convenience function to retrieve the model's spectral deviations."""
        if self.spectral_index_deviations is None:
            return None

        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        return self.hdvol * vmap(self.ht)(
            amplitude[self.pd] * self.spectral_index_deviations(p))

    def spectral_distribution(self, p):
        """Convenience function to retrieve the model's spectral
        distribution."""
        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)

        spectral_xi = p[f"{self.prefix}_spectral_index_xi"]
        spectral_index_mean = self.spectral_index_mean(p)
        spectral_index_fluc = self.spectral_index_fluctuations(p) * spectral_xi
        deviations = 0.0
        if self.spectral_index_deviations is not None:
            deviations = self.spectral_index_deviations(p)

        return self.hdvol*vmap(self.ht)(
            amplitude[self.pd] * (spectral_index_fluc*self._freqs + deviations)
        ) + spectral_index_mean * self._freqs


def build_default_mf_model(
    prefix: str,
    shape: tuple[int],
    distances: tuple[float],
    log_frequencies: Union[tuple[float], ArrayLike],
    reference_frequency_index: int,
    zero_mode_settings: Union[tuple, Callable],
    spatial_amplitude_settings: dict,
    spectral_index_settings: dict,
    spectral_amplitude_settings: Optional[dict] = None,
    deviations_settings: Optional[dict] = None,
    spatial_amplitude_model: str = "non_parametric",
    spectral_amplitude_model: str = "non_parametric",
    harmonic_type: str = 'fourier',
    dtype: type = jnp.float64,
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

    zero_mode_settings: tuple or callable
        Settings for the zero mode priors, by default a Gaussian.
            - Gaussian (default), (mean, std)

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

    fluct = 'fluctuations' if 'fluctuations' in spatial_amplitude_settings else 'scale' # FIXME: FIX WITH NORMAMP
    spatial_fluctuations = _build_distribution_or_default(
        spatial_amplitude_settings[fluct],
        f'{prefix}_spatial_fluctuations',
        lognormal_prior
    )

    spatial_amplitude = build_amplitude_model(
        grid,
        spatial_amplitude_settings,
        prefix=f'{prefix}_spatial',
        amplitude_model=spatial_amplitude_model)

    spectral_index_mean = _build_distribution_or_default(
        spectral_index_settings['mean'],
        f'{prefix}_spectral_index_mean',
        normal_prior
    )
    spectral_index_fluctuations = _build_distribution_or_default(
        spectral_index_settings['fluctuations'],
        f'{prefix}_spectral_index_fluctuations',
        lognormal_prior
    )

    spectral_amplitude = build_amplitude_model(
        grid,
        spectral_amplitude_settings,
        prefix=f'{prefix}_spectral',
        amplitude_model=spectral_amplitude_model)

    if spectral_amplitude is not None:
        logger.info("Both `spectral_amplitude` and `spectral_index` provided."
                    "\nThe fluctuations from `spectral_amplitude` model will "
                    "be ignored. The `spectral_index` fluctuations will be "
                    "used instead.")

    deviations_model = build_frequency_deviations_model(
        shape, log_frequencies, reference_frequency_index, deviations_settings,
        prefix=f'{prefix}_spectral')

    zero_mode = _build_distribution_or_default(
        zero_mode_settings,
        f'{prefix}_zero_mode',
        normal_prior
    )

    return CorrelatedMultiFrequencySky(
        prefix=prefix,
        relative_log_frequencies=jnp.array(
            log_frequencies) - log_frequencies[reference_frequency_index],
        zero_mode=zero_mode,
        spatial_fluctuations=spatial_fluctuations,
        spatial_amplitude=spatial_amplitude,
        spectral_index_mean=spectral_index_mean,
        spectral_index_fluctuations=spectral_index_fluctuations,
        spectral_amplitude=spectral_amplitude,
        spectral_index_deviations=deviations_model,
        dtype=dtype
    )
