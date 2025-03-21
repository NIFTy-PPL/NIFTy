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

from .frequency_deviations import build_frequency_deviations_model_with_degeneracies
from .mf_model_utils import (
    _build_distribution_or_default, build_normalized_amplitude_model)
from .spectral_behavior import (
    HarmonicLogSpectralBehavior, SingleHarmonicLogSpectralBehavior,
    SpectralIndex)
from .scaled_excitations import ScaledExcitations, build_scaled_excitations

from ..tree_math.vector import Vector
from .. import logger
from ..correlated_field import (
    make_grid, hartley, MaternAmplitude, NonParametricAmplitude)
from ..model import Model
from ..num.stats_distributions import lognormal_prior, normal_prior


class CorrelatedMultiFrequencySky(Model):
    """A model for generating a correlated multi-frequency sky map based on
    spatial and spectral correlation models.

    .. math ::
        sky = \\exp(F[A_spatial *
              i_\\mathrm{ref}(k, \\mu_\\mathrm{ref}) + A_spectral *
              (slope(k) * (\\mu-\\mu_\\mathrm{ref}) +
              GaussMarkovProcess(k, \\mu-\\mu_\\mathrm{ref})
              - AvgSlope[GaussMarkovProcess]
              )] + zero_mode)

    Parameters
    ----------
    relative_log_frequencies: Union[tuple[float], ArrayLike]
        The log_frequencies relative to the reference frequency:
        delta log(v) = log(v) - log(v_ref) = \\mu - \\mu_\\mathrm{ref}
    zero_model: jft.Model
        The model for the zero mode
    spatial_scaled_excitations: jft.Model
        Spatial excitations xi scaled by the fluctuations at reference frequency.
    spatial_amplitude: Union[MaternAmplitude, NonParametricAmplitude]
        Amplitude model for the spatial correlations.
    log_spectral_behavior: HarmonicLogSpectralBehavior
        The log spectral behavior of the model. See, e.g. `SpectralIndex`.
    spectral_amplitude: Optional[Union[MaternAmplitude,
        NonParametricAmplitude]]
        An optional amplitude model for the spectral correlations of the
        spectral_index field.
        If `None` the `spatial_amplitude` is used for the spectral
        correlations.
    spectral_index_deviations: Optional[jft.Model]
        A model capturing deviations from the spectral behavior of the
        spectral index model.
    log_ref_freq_mean_model: Optional[jft.Model]
        Optional model applied to the spatial part of the model in
        `nonlinearity` units. This can, for example, be used to tapper the
        spatial reference model.
    nonlinearity: Optional[jnp.callable]
        The nonlinearity to be applied to the multifrequency correlated
        field.
    """

    def __init__(
        self,
        zero_mode: Model,
        spatial_scaled_excitations: ScaledExcitations,
        spatial_amplitude: Union[MaternAmplitude, NonParametricAmplitude],
        log_spectral_behavior: HarmonicLogSpectralBehavior,
        spectral_amplitude: Optional[
            Union[MaternAmplitude, NonParametricAmplitude]] = None,
        spectral_index_deviations: Optional[Model] = None,
        log_ref_freq_mean_model: Optional[Model] = None,
        nonlinearity: Optional[Callable] = jnp.exp,
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
        self._hdvol = 1.0 / grid.total_volume
        self._pd = grid.harmonic_grid.power_distributor
        self._ht = partial(hartley, axes=tuple(range(len(grid.shape))))
        self._nonlinearity = nonlinearity

        self.zero_mode = zero_mode
        self.spatial_scaled_excitations = spatial_scaled_excitations
        self.spatial_amplitude = spatial_amplitude
        self.log_ref_freq_mean_model = log_ref_freq_mean_model

        self.log_spectral_behavior = log_spectral_behavior
        self.spectral_index_deviations = spectral_index_deviations
        self.spectral_amplitude = spectral_amplitude

        models = [
            self.zero_mode,
            self.spatial_scaled_excitations,
            self.spatial_amplitude,
            self.log_ref_freq_mean_model,
        ]

        logfreqs = self.log_spectral_behavior.relative_log_frequencies
        if logfreqs.shape[0] > 1:
            models += [
                self.spectral_index_deviations,
                self.log_spectral_behavior,
                self.spectral_amplitude,
            ]

        domain = reduce(
            lambda a, b: a | b,
            [(m.domain.tree if isinstance(m.domain, Vector) else m.domain)
             for m in models if m is not None]
        )

        if logfreqs.shape[0] == 1:
            self.spectral_amplitude = None
            self.spectral_index_distribution = None
            self.spectral_deviations_distribution = None
            self.spectral_distribution = None

        super().__init__(domain=domain)

    def __call__(self, p):
        if self.log_spectral_behavior.relative_log_frequencies.shape[0] == 1:
            return self.reference_frequency_distribution(p)[None, ...]

        zm = self.zero_mode(p)
        spat_amplitude = self.spatial_amplitude(p)
        spat_amplitude = spat_amplitude.at[0].set(0.0)
        spatial_reference = self.spatial_scaled_excitations(p)

        if (self.spectral_index_deviations is None and
                isinstance(self.log_spectral_behavior,
                           SingleHarmonicLogSpectralBehavior)):
            """Apply method for single spectral behavior and no spectral
            deviations. In this case, we can save some harmonic transforms.

            Implements:
            .. math::
                    sky = \\nonlinearity(
                    HT[A_spatial * io(k, \\mu_0)] +
                    (HT[A_spectral * spectral_fluctuations(k)] + spectral_mean)
                    * (\\mu-\\mu_0)
                    + zero_mode)
            where :math:`F` is the Fourier transform,
            :math:`k` is the spatial frequency index,
            :math:`\\mu` is the log spectral frequency index,
            and `slope` represents the spectral index.
            """
            spectral_index_mean = self.log_spectral_behavior.mean(p)

            if self.spectral_amplitude is None:
                spec_amplitude = spat_amplitude
            else:
                spec_amplitude = self.spectral_amplitude(p)
                spec_amplitude = spec_amplitude.at[0].set(0.0)

            correlated_spatial_reference = self._hdvol*self._ht(
                spat_amplitude[self._pd]*spatial_reference)
            correlated_spectral_index = self._hdvol * self._ht(
                spec_amplitude[self._pd]*self.log_spectral_behavior.fluctuations(p))

            if self.log_ref_freq_mean_model is None:
                return self._nonlinearity(
                    correlated_spatial_reference +
                    (correlated_spectral_index + spectral_index_mean) *
                    self.log_spectral_behavior.relative_log_frequencies +
                    zm)
            else:
                return self._nonlinearity(
                    self.log_ref_freq_mean_model(p) +
                    correlated_spatial_reference +
                    (correlated_spectral_index + spectral_index_mean) *
                    self.log_spectral_behavior.relative_log_frequencies +
                    zm)

        else:
            """Apply method for a general `HarmonicLogSpectralBehavior` with or
            without spectral index deviations.

            Implements:
            .. math::
                    sky = \\nonlinearity(
                    F[A_spatial*io(k, \\mu_0) +
                      A_spectral*(
                        spectral_fluctuations(k)*(\\mu-\\mu_0)
                        + GaussMarkovProcess(k, \\mu-\\mu_0)
                        - AvgSlope[GaussMarkovProcess]*(\\mu-\\mu_0)
                      )] + spectral_mean*(\\mu-\\mu_0) + zero_mode)
            where :math:`F` is the Fourier transform,
            :math:`k` is the spatial frequency index,
            :math:`\\mu` is the log spectral frequency index,
            and `slope` represents the spectral index.
            """
            if self.spectral_index_deviations is None:
                spectral_terms = self.log_spectral_behavior.fluctuations_with_frequencies(
                    p)
            else:
                spectral_deviations = self.spectral_index_deviations(p)
                spectral_terms = (
                    self.log_spectral_behavior.fluctuations_with_frequencies(p) +
                    self.log_spectral_behavior.remove_degeneracy_of_spectral_deviations(
                        spectral_deviations)
                )

            if self.spectral_amplitude is None:
                cf_values = self._hdvol*vmap(self._ht)(
                    spat_amplitude[self._pd]*(spatial_reference+spectral_terms)
                )
            else:
                spec_amplitude = self.spectral_amplitude(p)
                spec_amplitude = spec_amplitude.at[0].set(0.0)

                cf_values = self._hdvol*vmap(self._ht)(
                    spat_amplitude[self._pd]*spatial_reference +
                    spec_amplitude[self._pd]*spectral_terms)

            spectral_mean = self.log_spectral_behavior.mean_with_frequencies(p)
            if self.log_ref_freq_mean_model is None:
                return self._nonlinearity(cf_values + spectral_mean + zm)
            else:
                return self._nonlinearity(
                    self.log_ref_freq_mean_model(p)+cf_values+spectral_mean+zm)

    def reference_frequency_distribution(self, p):
        """Convenience method to retrieve the model's spatial distribution at
        the reference frequency."""
        amplitude = self.spatial_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        if self.log_ref_freq_mean_model is None:
            return self._nonlinearity(
                self._hdvol*self._ht(
                    amplitude[self._pd] * self.spatial_scaled_excitations(p))
                + self.zero_mode(p))
        else:
            return self._nonlinearity(
                self.log_ref_freq_mean_model(p) +
                self._hdvol * self._ht(
                    amplitude[self._pd] * self.spatial_scaled_excitations(p))
                + self.zero_mode(p))

    def spectral_index_distribution(self, p, order: int = 1):
        """Convenience method to retrieve the model's spectral index. Or the
        higher order terms which is controlled by the order parameter."""

        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        spectral_mean = self.log_spectral_behavior.mean(p)
        spectral_fluc = self.log_spectral_behavior.fluctuations(p)

        if isinstance(self.log_spectral_behavior, SingleHarmonicLogSpectralBehavior):
            return (self._hdvol*self._ht(
                amplitude[self._pd] * spectral_fluc
            ) + spectral_mean)

        return jnp.array([
            (self._hdvol*self._ht(amplitude[self._pd]*spf)) + spm
            for spm, spf in zip(spectral_mean[:order], spectral_fluc[:order])
        ])

    def spectral_deviations_distribution(self, p):
        """Convenience method to retrieve the model's spectral deviations."""
        if self.spectral_index_deviations is None:
            return None

        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        spectral_deviations = self.log_spectral_behavior.remove_degeneracy_of_spectral_deviations(
            self.spectral_index_deviations(p))
        return self._hdvol * vmap(self._ht)(
            amplitude[self._pd] * spectral_deviations)

    def spectral_distribution(self, p):
        """Convenience method to retrieve the model's spectral
        distribution."""
        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)

        deviations = 0.0
        if self.spectral_index_deviations is not None:
            deviations = self.spectral_index_deviations(p)

        spectral_fluc = self.log_spectral_behavior.fluctuations_with_frequencies(
            p)
        spectral_mean = self.log_spectral_behavior.mean_with_frequencies(p)
        return self._hdvol*vmap(self._ht)(
            amplitude[self._pd]*(spectral_fluc + deviations)
        ) + spectral_mean

    def _get_deviations_at_relative_log_freqency(
        self,
        p,
        relative_log_frequency
    ):
        """Convenience method to retrieve the model's deviations at
        a given relative log frequency."""
        # TODO: deviations could be recalculated at
        # given relative_log_frequency instead of interp
        if self.spectral_index_deviations is None:
            return 0.0

        deviations = self.spectral_index_deviations(p)

        def _interpolate_1d(times_series, times, target_time):
            return jnp.interp(target_time, times, times_series)

        _interp = partial(
            _interpolate_1d,
            times=self.log_spectral_behavior.relative_log_frequencies.reshape(
                -1),
            target_time=relative_log_frequency
        )

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
        """Convenience method to retrieve the model's spectral
        distribution at a given relative frequency."""
        if self.spectral_amplitude is None:
            amplitude = self.spatial_amplitude(p)
        else:
            amplitude = self.spectral_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)

        spectral_index_mean = self.log_spectral_behavior.mean_with_frequencies(
            p)
        spectral_index_fluc = self.log_spectral_behavior.fluctuations_with_frequencies(
            p)

        deviations = self._get_deviations_at_relative_log_freqency(
            p, relative_log_frequency)

        return self._hdvol*self._ht(
            amplitude[self._pd] * (spectral_index_fluc + deviations)
        ) + spectral_index_mean

    def get_distribution_at_relative_log_frequency(
        self,
        p,
        relative_log_frequency
    ):
        """Convenience method to retrieve the model's distribution
        at a given relative frequency."""
        spatial_distr = self.reference_frequency_distribution(p)
        spec_dist = self.get_spectral_distribution_at_relative_log_frequency(
            p, relative_log_frequency)
        # FIXME: this only works for the default nonlinearity
        return spatial_distr * self._nonlinearity(spec_dist)

    def reference_frequency_correlated_field(self, p):
        """Convenience method to retrieve the model's spatial distribution
        perturbations at the reference frequency."""
        amplitude = self.spatial_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        return self._nonlinearity(self._hdvol*self._ht(
            amplitude[self._pd] * self.spatial_scaled_excitations(p)
        ) + self.zero_mode(p))

    def reference_frequency_mean_distribution(self, p):
        """Convenience method to retrieve the model's mean spatial
        distribution at the reference frequency."""
        if self.log_ref_freq_mean_model is None:
            return 1.
        return self._nonlinearity(self.log_ref_freq_mean_model(p))


def build_default_mf_model(
    prefix: str,
    shape: tuple[int],
    distances: tuple[float],
    log_frequencies: Union[tuple[float], ArrayLike],
    reference_frequency_index: int,
    zero_mode_settings: Union[tuple, Callable],
    spatial_amplitude_settings: dict,
    spectral_index_settings: dict,
    spectral_index_mean: Optional[Model] = None,
    spectral_index_fluctuations: Optional[Model] = None,
    spectral_amplitude_settings: Optional[dict] = None,
    deviations_settings: Optional[dict] = None,
    log_reference_frequency_mean_model: Optional[Model] = None,
    spatial_amplitude_model: str = "non_parametric",
    spectral_amplitude_model: str = "non_parametric",
    harmonic_type: str = 'fourier',
    nonlinearity: callable = jnp.exp,
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

    nonlinearity: callable
        The nonlinearity of the multifrequency model, exp (default).

    Returns
    -------
    model: CorrelatedMultiFrequencySky
        The multi-frequency sky model
    """

    grid = make_grid(shape, distances, harmonic_type)

    # FIXME: FIX WITH NORMAMP
    fluct = 'fluctuations' if 'fluctuations' in spatial_amplitude_settings else 'scale'
    spatial_fluctuations = build_scaled_excitations(
        f'{prefix}_spatial',
        fluctuation_settings=spatial_amplitude_settings[fluct],
        shape=shape,
    )
    spatial_amplitude = build_normalized_amplitude_model(
        grid,
        spatial_amplitude_settings,
        prefix=f'{prefix}_spatial',
        amplitude_model=spatial_amplitude_model)

    spectral_amplitude = build_normalized_amplitude_model(
        grid,
        spectral_amplitude_settings,
        prefix=f'{prefix}_spectral',
        amplitude_model=spectral_amplitude_model)
    if spectral_index_mean is None:
        spectral_index_mean = _build_distribution_or_default(
            spectral_index_settings['mean'],
            f'{prefix}_spectral_index_mean',
            normal_prior
        )
    if spectral_index_fluctuations is None:
        spectral_index_fluctuations = build_scaled_excitations(
            prefix=f'{prefix}_spectral_index',
            fluctuation_settings=spectral_index_settings['fluctuations'],
            shape=shape,
        )

    log_spectral_behavior = SpectralIndex(
        log_frequencies=log_frequencies,
        mean=spectral_index_mean,
        spectral_scaled_excitations=spectral_index_fluctuations,
        reference_frequency_index=reference_frequency_index,
    )

    if spectral_amplitude is not None:
        logger.info("Both `spectral_amplitude` and `spectral_index` provided."
                    "\nThe fluctuations from `spectral_amplitude` model will "
                    "be ignored. The `spectral_index` fluctuations will be "
                    "used instead.")

    deviations_model = build_frequency_deviations_model_with_degeneracies(
        shape, log_frequencies, reference_frequency_index, deviations_settings,
        prefix=f'{prefix}_spectral')

    zero_mode = _build_distribution_or_default(
        zero_mode_settings,
        f'{prefix}_zero_mode',
        normal_prior
    )

    sky = CorrelatedMultiFrequencySky(
        zero_mode=zero_mode,
        spatial_scaled_excitations=spatial_fluctuations,
        spatial_amplitude=spatial_amplitude,
        log_spectral_behavior=log_spectral_behavior,
        spectral_amplitude=spectral_amplitude,
        spectral_index_deviations=deviations_model,
        log_ref_freq_mean_model=log_reference_frequency_mean_model,
        nonlinearity=nonlinearity, )

    return add_prefix(sky, prefix)


def add_xi_keys_properties(object: CorrelatedMultiFrequencySky):

    @property
    def spatial_xi_key(self: CorrelatedMultiFrequencySky):
        sc_excitations: ScaledExcitations = self.spatial_scaled_excitations
        return next(iter(sc_excitations.latent_space_xi.domain.keys()))

    @property
    def spectral_xi_key(self: CorrelatedMultiFrequencySky):
        spectral: HarmonicLogSpectralBehavior = self.log_spectral_behavior
        sc_excitations: ScaledExcitations = spectral.spectral_scaled_excitations
        return next(iter(sc_excitations.latent_space_xi.domain.keys()))

    @property
    def spectral_deviations_xi_key(self: CorrelatedMultiFrequencySky):
        deviations: Model = self.spectral_index_deviations
        for k in deviations.domain.keys():
            if "xi" in k:
                return k
        raise ValueError("No xi key found in deviations model.")

    setattr(object, "spatial_xi_key", spatial_xi_key)
    setattr(object, "spectral_xi_key", spectral_xi_key)
    if object.spectral_index_deviations is not None:
        setattr(object,
                "spectral_deviations_xi_key",
                spectral_deviations_xi_key)
    return object


def add_prefix(
    object: CorrelatedMultiFrequencySky,
    prefix_key: str
):
    setattr(object, "_prefix", prefix_key)
    return object