# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig,
# Vincent Eberle, Philipp Frank, Vishal Johnson,
# Jakob Roth, Margret Westerkamp

from functools import reduce, partial
from typing import Optional, Union, Callable

import jax.numpy as jnp
import numpy as np
from jax import vmap
from numpy.typing import ArrayLike

from ..tree_math.vector import Vector
from .frequency_deviations import build_frequency_deviations_model
from .mf_model_utils import (
    _build_distribution_or_default, build_normalized_amplitude_model)
from .. import ShapeWithDtype, logger
from ..correlated_field import (
    _make_grid, hartley, MaternAmplitude, NonParametricAmplitude)
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
        relative_log_frequencies: Union[tuple[float], ArrayLike],
        zero_mode: Model,
        spatial_fluctuations: Model,
        spatial_amplitude: Union[MaternAmplitude, NonParametricAmplitude],
        spectral_index_mean: Model,
        spectral_index_fluctuations: Model,
        spectral_amplitude: Optional[
            Union[MaternAmplitude, NonParametricAmplitude]] = None,
        spectral_index_deviations: Optional[Model] = None,
        log_ref_freq_mean_model: Optional[Model] = None,
        nonlinearity: Optional[Callable] = jnp.exp,
        dtype: type = jnp.float64,
    ):

        # The amplitudes supplied need to be normalized, as both the spatial
        # and the spectral fluctuations are applied directly in the call to
        # avoid degeneracies.
        if spatial_amplitude.fluctuations is not None:
            raise ValueError(
                "Spatial amplitude must be normalized."
                "It is not allowed to have `fluctuations`."
            )

        if spectral_amplitude is not None:
            if spectral_amplitude.fluctuations is not None:
                raise ValueError(
                    "Spectral amplitude must be normalized."
                    "It is not allowed to have `fluctuations`."
                )

        grid = spatial_amplitude.grid
        slicing_tuple = (slice(None),) + (None,) * len(grid.shape)
        self._prefix = prefix
        self._freqs = np.array(relative_log_frequencies)[slicing_tuple]
        self._hdvol = 1.0 / grid.total_volume
        self._pd = grid.harmonic_grid.power_distributor
        self._ht = partial(hartley, axes=tuple(range(len(grid.shape))))
        self._nonlinearity = nonlinearity

        self.zero_mode = zero_mode
        self._spatial_fluctuations = spatial_fluctuations
        self.spatial_amplitude = spatial_amplitude
        self._spectral_index_mean = spectral_index_mean
        self._spectral_index_fluctuations = spectral_index_fluctuations
        self._spectral_index_deviations = spectral_index_deviations
        self.spectral_amplitude = spectral_amplitude
        self.log_ref_freq_mean_model = log_ref_freq_mean_model

        models = [
            self.zero_mode,
            self._spatial_fluctuations,
            self.spatial_amplitude,
            self.log_ref_freq_mean_model
        ]

        if self._freqs.shape[0] > 1:
            models += [
                self._spectral_index_mean,
                self._spectral_index_fluctuations,
                self._spectral_index_deviations,
                self.spectral_amplitude,
            ]

        domain = reduce(
            lambda a, b: a | b,
            [(m.domain.tree if isinstance(m.domain, Vector) else m.domain)
             for m in models if m is not None]
        )

        domain[f"{self._prefix}_spatial_xi"] = ShapeWithDtype(
            grid.shape, dtype)
        domain[f"{self._prefix}_spectral_index_xi"] = ShapeWithDtype(
            grid.shape, dtype)

        if self._freqs.shape[0] == 1:
            domain.pop(f"{self._prefix}_spectral_index_xi")
            self.spectral_amplitude = None
            self.spectral_index_distribution = None
            self.spectral_deviations_distribution = None
            self.spectral_distribution = None

        super().__init__(domain=domain)

    def __call__(self, p):
        if self._freqs.shape[0] == 1:
            return self.reference_frequency_distribution(p)[None, ...]
        if self._spectral_index_deviations is not None:
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
            distributed_spatial_amplitude = spat_amplitude[self._pd]

            spatial_xis = p[f"{self._prefix}_spatial_xi"]
            spectral_xis = p[f"{self._prefix}_spectral_index_xi"]
            spatial_reference = self._spatial_fluctuations(p) * spatial_xis
            spectral_index_mean = self._spectral_index_mean(p)
            spectral_terms = (self._spectral_index_fluctuations(p) *
                              spectral_xis * self._freqs +
                              self._spectral_index_deviations(p))

            if self.spectral_amplitude is None:
                cf_values = self._hdvol*vmap(self._ht)(
                    distributed_spatial_amplitude *
                    (spatial_reference + spectral_terms)
                )
            else:
                spec_amplitude = self.spectral_amplitude(p)
                spec_amplitude = spec_amplitude.at[0].set(0.0)
                distributed_spectral_amplitude = spec_amplitude[self._pd]

                cf_values = self._hdvol*vmap(self._ht)(
                    distributed_spatial_amplitude * spatial_reference +
                    distributed_spectral_amplitude * spectral_terms)

            if self.log_ref_freq_mean_model is None:
                return self._nonlinearity(
                    cf_values + spectral_index_mean*self._freqs + zm)
            else:
                return self._nonlinearity(
                    self.log_ref_freq_mean_model(p) +
                    spectral_index_mean * self._freqs +
                    zm)

        else:
            """
            Apply method of the model without spectral
            index deviations.
            Implements:
            .. math::
                    sky = \\nonlinearity(
                    F[A_spatial * io(k, \\mu_0)] +
                    (F[A_spectral * slope_fluctuations(k)] + slope_mean) *
                      (\\mu-\\mu_0)
                    + zero_mode)
            where :math:`F` is the Fourier transform,
            :math:`k` is the spatial frequency index,
            :math:`\\mu` is the log spectral frequency index,
            and `slope` represents the spectral index.
            """
            zm = self.zero_mode(p)
            spat_amplitude = self.spatial_amplitude(p)
            spat_amplitude = spat_amplitude.at[0].set(0.0)

            spatial_xis = p[f"{self._prefix}_spatial_xi"]
            spectral_xis = p[f"{self._prefix}_spectral_index_xi"]
            spatial_reference = self._spatial_fluctuations(p) * spatial_xis
            spectral_index_mean = self._spectral_index_mean(p)

            if self.spectral_amplitude is None:
                spec_amplitude = spat_amplitude
            else:
                spec_amplitude = self.spectral_amplitude(p)
                spec_amplitude = spec_amplitude.at[0].set(0.0)

            correlated_spatial_reference = self._hdvol*self._ht(
                spat_amplitude[self._pd]*spatial_reference)
            correlated_spectral_index = self._hdvol * self._ht(
                spec_amplitude[self._pd] *
                self._spectral_index_fluctuations(p) *
                spectral_xis)

            if self.log_ref_freq_mean_model is None:
                return self._nonlinearity(
                    correlated_spatial_reference +
                    correlated_spectral_index +
                    spectral_index_mean*self._freqs +
                    zm)
            else:
                return self._nonlinearity(
                    self.log_ref_freq_mean_model(p) +
                    correlated_spatial_reference +
                    correlated_spectral_index +
                    spectral_index_mean * self._freqs +
                    zm)

    def reference_frequency_distribution(self, p):
        """Convenience function to retrieve the model's spatial distribution at
        the reference frequency."""
        amplitude = self.spatial_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        spatial_xi = p[f"{self._prefix}_spatial_xi"]
        if self.log_ref_freq_mean_model is None:
            return self._nonlinearity(
                self._hdvol*self._ht(
                    amplitude[self._pd] * self._spatial_fluctuations(p) *
                    spatial_xi)
                + self.zero_mode(p)
            )
        else:
            return self._nonlinearity(
                self.log_ref_freq_mean_model(p) +
                self._hdvol * self._ht(
                    amplitude[self._pd] * self._spatial_fluctuations(p) *
                    spatial_xi)
            )

    def spectral_index_distribution(self, p):
        """Convenience function to retrieve the model's spectral index."""
        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        spectral_xi = p[f"{self._prefix}_spectral_index_xi"]
        amplitude = amplitude.at[0].set(0.0)
        spectral_index_mean = self._spectral_index_mean(p)

        return (self._hdvol*self._ht(amplitude[self._pd] *
                                     self._spectral_index_fluctuations(p) *
                                     spectral_xi)
                + spectral_index_mean)

    def spectral_deviations_distribution(self, p):
        """Convenience function to retrieve the model's spectral deviations."""
        if self._spectral_index_deviations is None:
            return None

        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        return self._hdvol * vmap(self._ht)(
            amplitude[self._pd] * self._spectral_index_deviations(p))

    def spectral_distribution(self, p):
        """Convenience function to retrieve the model's spectral
        distribution."""
        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)

        spectral_xi = p[f"{self._prefix}_spectral_index_xi"]
        spectral_index_mean = self._spectral_index_mean(p)
        spectral_index_fluc = self._spectral_index_fluctuations(
            p) * spectral_xi
        deviations = 0.0
        if self._spectral_index_deviations is not None:
            deviations = self._spectral_index_deviations(p)

        return self._hdvol*vmap(self._ht)(
            amplitude[self._pd] * (spectral_index_fluc *
                                   self._freqs + deviations)
        ) + spectral_index_mean * self._freqs

    def _get_deviations_at_relative_log_freqency(
        self,
        p,
        relative_log_frequency
    ):
        """Convenience function to retrieve the model's deviations at
        a given relative log frequency."""
        # TODO: deviations could be recalculated at
        # given relative_log_frequency instead of interp
        if self._spectral_index_deviations is None:
            return 0.0

        deviations = self._spectral_index_deviations(p)

        def _interpolate_1d(times_series, times, target_time):
            return jnp.interp(target_time, times, times_series)

        _interp = partial(_interpolate_1d,
                          times=self._freqs.reshape(-1),
                          target_time=relative_log_frequency)

        if len(deviations.shape) == 2:
            return vmap(_interp)(deviations.T)

        elif len(deviations.shape) == 3:
            _mapped_interp = vmap(vmap(_interp, in_axes=0), in_axes=0)
            return _mapped_interp(np.moveaxis(deviations, 0, -1))

        else:
            raise NotImplementedError("Deviations interpolation works only "
                                      "for 1 or 2 spatial dimensions.")

    def get_spectral_distribution_at_relative_log_frequency(
        self,
        p,
        relative_log_frequency
    ):
        """Convenience function to retrieve the model's spectral
        distribution at a given relative frequency."""
        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)

        spectral_xi = p[f"{self._prefix}_spectral_index_xi"]
        spectral_index_mean = self._spectral_index_mean(p)
        spectral_index_fluc = self._spectral_index_fluctuations(
            p) * spectral_xi

        deviations = self._get_deviations_at_relative_log_freqency(
            p, relative_log_frequency)

        return self._hdvol*self._ht(
            amplitude[self._pd] * (spectral_index_fluc *
                                   relative_log_frequency + deviations)
        ) + spectral_index_mean * relative_log_frequency

    def get_distribution_at_relative_log_frequency(
        self,
        p,
        relative_log_frequency
    ):
        """Convenience function to retrieve the model's distribution
        at a given relative frequency."""
        spatial_distr = self.reference_frequency_distribution(p)
        spec_dist = self.get_spectral_distribution_at_relative_log_frequency(
            p,
            relative_log_frequency
        )
        # FIXME: this only works for the default nonlinearity
        return spatial_distr * self._nonlinearity(spec_dist)


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
    log_reference_frequency_mean_model: Optional[Model] = None,
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

    log_reference_frequency_mean_model: Model
        Model for the distribution of the log mean of the
        reference frequency.

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

    dtype: type
        The type of the parameters.

    Returns
    -------
    model: CorrelatedMultiFrequencySky
        The multi-frequency sky model
    """

    grid = _make_grid(shape, distances, harmonic_type)

    # FIXME: FIX WITH NORMAMP
    fluct = 'fluctuations' if 'fluctuations' in spatial_amplitude_settings else 'scale'
    spatial_fluctuations = _build_distribution_or_default(
        spatial_amplitude_settings[fluct],
        f'{prefix}_spatial_fluctuations',
        lognormal_prior
    )

    spatial_amplitude = build_normalized_amplitude_model(
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

    spectral_amplitude = build_normalized_amplitude_model(
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
        log_ref_freq_mean_model=log_reference_frequency_mean_model,
        dtype=dtype
    )
