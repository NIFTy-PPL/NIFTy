from functools import reduce, partial
from typing import Optional

import jax.numpy as jnp

from ..num.stats_distributions import lognormal_prior, normal_prior
from ..model import Model
from ..gauss_markov import WienerProcess
from ..correlated_field import (
    matern_amplitude, _make_grid, hartley, RegularCartesianGrid,
    WrappedCall)

from .frequency_deviations import (
    FrequencyDeviations, build_frequency_devations_from_1d_process)


def _check_demands(model_name, kwargs, demands):
    for key in demands:
        assert key in kwargs, (f'{key} not in {model_name}.\n'
                               f'Provide settings for {key}')


def _set_default_kwargs(kwargs, defaults):
    """
    Update kwargs with default values if not already present.

    Args:
    kwargs (dict): The keyword arguments dictionary.
    defaults (dict): A dictionary of default key-value pairs.

    Returns:
    dict: Updated kwargs dictionary with default values.
    """
    return {**defaults, **kwargs}


def _set_default_or_callable(key: str, kwargs: dict, default: callable):
    '''Set either default distribution or the callable'''

    if callable(kwargs[key]):
        # TODO: do a check here that it is a valid distribution.
        return kwargs[key]

    return default(*kwargs[key])


def _build_distribution(
        prefix: str, key: str, kwargs: dict, default: callable, shape=()):
    return WrappedCall(
        _set_default_or_callable(key, kwargs, default),
        name=f'{prefix}_{key}',
        white_init=True)


def build_zero_mode_model(
    prefix: str,
    zero_mode_settings: dict,
) -> Model:

    zm_name = f'{prefix}_zero_mode'
    _check_demands(zm_name, zero_mode_settings, demands={'mean', 'deviations'})

    zero_mode_mean = zero_mode_settings['mean']
    zero_mode_deviations = _build_distribution(
        zm_name, 'deviations', zero_mode_settings, default=normal_prior)

    return Model(
        lambda x: zero_mode_mean + zero_mode_deviations(x),
        init=zero_mode_deviations.init)


def build_amplitude_model(
    prefix: str,
    grid_2d: RegularCartesianGrid,
    amplitude_settings: dict,
) -> Model:

    # FIXME: Need support for correlated_field

    amplitude_name = f'{prefix}_amplitude'
    amplitude_settings = _set_default_kwargs(
        amplitude_settings, defaults=dict(renormalize_amplitude=False))
    _check_demands(amplitude_name,
                   amplitude_settings,
                   demands={'scale', 'cutoff', 'loglogslope'})

    return matern_amplitude(
        grid_2d,
        scale=_set_default_or_callable(
            'scale', amplitude_settings, default=lognormal_prior),
        cutoff=_set_default_or_callable(
            'cutoff', amplitude_settings, default=lognormal_prior),
        loglogslope=_set_default_or_callable(
            'loglogslope', amplitude_settings, default=normal_prior),
        renormalize_amplitude=amplitude_settings['renormalize_amplitude'],
        prefix=amplitude_name)


def build_spatial_xi_at_reference_frequency(
    prefix: str,
    shape_2d: tuple[int],
) -> Model:

    return WrappedCall(
        normal_prior(0.0, 1.0),
        shape=shape_2d,
        name=f'{prefix}_spatial_xi',
        white_init=True
    )


def build_frequency_slope_model(
    prefix: str,
    shape_2d: tuple[int],
    slope_settings: dict,
):
    slope_name = f'{prefix}_slope'
    _check_demands(slope_name, slope_settings, demands={'mean'})

    slope_mean = _build_distribution(
        slope_name, 'mean', slope_settings, default=normal_prior)
    slope_xi = _build_distribution(
        slope_name, 'xi', dict(xi=(0.0, 1.0)), normal_prior, shape=shape_2d)

    return Model(
        lambda x: slope_mean(x)*slope_xi(x),
        domain=slope_mean.domain | slope_xi.domain)


def build_deviations_model(
    prefix: str,
    shape_2d: tuple[int],
    log_frequencies: tuple[float],
    reference_frequency: int,
    deviations_settings: Optional[dict],
) -> FrequencyDeviations:

    if deviations_settings is None:
        return None

    dev_name = f'{prefix}_deviations'
    process_name = deviations_settings.get('process', 'wiener').lower()

    # FIXME: Need more processes
    if process_name in {'wiener', 'wiener_process'}:
        _check_demands(dev_name, deviations_settings, demands={'sigma'})
        flexibility = _build_distribution(
            dev_name, 'sigma', deviations_settings, default=lognormal_prior
        )
        process = WienerProcess(
            x0=0,
            sigma=flexibility,
            dt=log_frequencies[1:]-log_frequencies[0],
            name=f'{dev_name}_wp',
        )
    else:
        raise NotImplementedError(f'{process_name} not implemented.')

    return build_frequency_devations_from_1d_process(
        process, shape_2d, log_frequencies, reference_frequency)


class MfModel(Model):
    def __init__(
        self,
        grid_2d: RegularCartesianGrid,
        log_relative_frequencies: tuple[float],

        zero_mode: Model,
        spatial_amplitude: Model,
        spatial_xi_at_reference_frequency: Model,
        spectral_index_mean: Model,
        spectral_index_deviations: Optional[FrequencyDeviations] = None,
    ):
        # FIXME: Matteo, you know what to do with hdvol and the power_distributor?
        self.hdvol = 1.0 / grid_2d.total_volume
        self.pd = grid_2d.harmonic_grid.power_distributor
        self.ht = partial(hartley, axes=(0, 1))
        self.freqs = jnp.array(log_relative_frequencies)[:, None, None]

        self.zero_mode = zero_mode
        self.spatial_amplitude = spatial_amplitude
        self.spatial_xi = spatial_xi_at_reference_frequency
        self.spectral_index_mean = spectral_index_mean
        self.spectral_index_deviations = spectral_index_deviations

        self.apply = self._build_apply()

        models = [zero_mode, spatial_amplitude, spatial_xi_at_reference_frequency,
                  spectral_index_mean, spectral_index_deviations]
        domain = reduce(
            lambda a, b: a | b, [m.domain for m in models if m is not None]
        )

        super().__init__(self.apply, domain=domain)

    def _build_apply(self):
        if self.spectral_index_deviations is not None:

            def apply_with_deviations(p):
                zm = self.zero_mode(p)

                amplitude = self.spatial_amplitude(p)
                amplitude = amplitude.at[0].set(0.0)
                spatial_xi = self.spatial_xi(p)

                slope = self.spectral_index_mean(p)

                deviations = self.spectral_index_deviations(p)

                # FIXME, TODO: One can probably vmap over the log_frequencies
                # FIXME: jax.scan jax.for_i, probably
                # Something better with fourier trafo
                return jnp.exp(jnp.array(
                    [self.hdvol * self.ht(x)+zm for x in
                     amplitude[self.pd]*(spatial_xi + slope*self.freqs + deviations)]
                ))

            return apply_with_deviations

        def apply_without_deviations(p):
            zm = self.zero_mode(p)

            amplitude = self.spatial_amplitude(p)
            amplitude = amplitude.at[0].set(0.0)

            spatial_xi = self.spatial_xi(p)

            slope = self.spectral_index_mean(p)

            cfio = self.hdvol*self.ht(amplitude[self.pd]*spatial_xi)
            cfsl = self.hdvol*self.ht(amplitude[self.pd]*slope)
            return jnp.exp(cfio+zm + cfsl*self.freqs)

        return apply_without_deviations


def build_mf_model(
    prefix: str,
    shape_2d: tuple[int],
    distances_2d: tuple[float],
    log_frequencies: tuple[float],
    reference_frequency: int,

    zero_mode_settings: dict,
    amplitude_settings: dict,
    slope_settings: dict,
    deviations_settings: Optional[dict] = None,

    harmonic_type: str = 'fourier',
):
    '''Build multi frequency model.
        f = exp( F * A * (
              io(k, l0) +
              slope(k) * (l-l0) +
              GaussMarkovProcess(k, l-l0) (3d) - MeanSlope(GMP) (2d)
        ) * (offset_mean + deviations) )

    Parameters
    ----------
    prefix
    shape_2d: the shape of the spatial_amplitude domain (2d)
    distances_2d: the distances of the spatial_amplitude domain (2d)
    log_frequencies: the log log_frequencies
    reference_frequency: the identification number of the referency frequency

    zero_mode_settings:
        - mean: zero mode mean
        - deviations: callable or parameters for default (normal_prior)

    amplitude_settings:
        - scale: callable or parameters for default (lognormal_prior)
        - cutoff: callable or parameters for default (lognormal_prior)
        - loglogslope: callable or parameters for default (normal_prior)

    slope_settings:
        - mean: callable or parameters for default (normal_prior)

    deviations_settings: If none deviations are not build.
        - process: wiener (default)
    '''

    grid_2d = _make_grid(shape_2d, distances_2d, harmonic_type)

    zero_mode_model = build_zero_mode_model(prefix, zero_mode_settings)
    spatial_model = build_amplitude_model(prefix, grid_2d, amplitude_settings)
    spatial_xi_at_reference_frequency = build_spatial_xi_at_reference_frequency(
        prefix, shape_2d)
    frequency_slope_model = build_frequency_slope_model(
        prefix, shape_2d, slope_settings)
    deviations_model = build_deviations_model(
        prefix, shape_2d, log_frequencies, reference_frequency, deviations_settings)

    return MfModel(
        grid_2d=grid_2d,

        log_relative_frequencies=jnp.array(
            log_frequencies) - log_frequencies[reference_frequency],
        zero_mode=zero_mode_model,
        spatial_amplitude=spatial_model,
        spatial_xi_at_reference_frequency=spatial_xi_at_reference_frequency,
        spectral_index_mean=frequency_slope_model,
        spectral_index_deviations=deviations_model
    )
