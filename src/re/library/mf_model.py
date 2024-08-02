from functools import reduce, partial
from typing import Optional, Union

import jax.numpy as jnp
from jax import vmap
from numpy.typing import ArrayLike

from .. import ShapeWithDtype
from ..num.stats_distributions import lognormal_prior, normal_prior
from ..model import Model, LazyModel
from ..gauss_markov import build_wiener_process
from ..correlated_field import (
    matern_amplitude, _make_grid, hartley, RegularCartesianGrid,
    WrappedCall)

from .frequency_deviations import FrequencyDeviations


def _check_demands(model_name, kwargs, demands):
    for key in demands:
        assert key in kwargs, (f'{key} not in {model_name}.\n'
                               f'Provide settings for {key}')


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
        shape=shape,
        white_init=True)


def _build_distribution_or_default(arg: Union[callable, tuple],
                                   key: str,
                                   default: callable,
                                   shape: tuple = (),
                                   dtype=None):
    return WrappedCall(
            arg if callable(arg) else default(*arg),
            name=key,
            shape=shape,
            dtype=dtype,
            white_init=True)


def build_amplitude_model(
    prefix: str,
    grid_2d: RegularCartesianGrid,
    amplitude_settings: dict,
) -> Model:
    # TODO: Need support for correlated_field
    amplitude_name = f'{prefix}_amplitude_'
    amplitude_settings = {**amplitude_settings,
                          **dict(renormalize_amplitude=False)}
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


def build_deviations_model(
    prefix: str,
    shape_2d: tuple[int],
    log_frequencies: Union[tuple[float], ArrayLike],
    reference_frequency_index: int,
    deviations_settings: Optional[dict],
) -> FrequencyDeviations:

    if deviations_settings is None:
        return None

    dev_name = f'{prefix}_deviations'
    process_name = deviations_settings.get('process', 'wiener').lower()

    # FIXME: Need more processes
    if process_name in {'wiener', 'wiener_process'}:
        _check_demands(dev_name, deviations_settings, demands={'sigma'})
        sigma = _build_distribution(dev_name, 'sigma',
                                    deviations_settings,
                                    default=lognormal_prior)

        domain_key = f'{dev_name}_wp'
        process = build_wiener_process(
            x0=jnp.zeros(shape_2d),  # 0,
            sigma=sigma,
            dt=log_frequencies[1:]-log_frequencies[0],
            name=domain_key,
        )
    else:
        raise NotImplementedError(f'{process_name} not implemented.')

    return FrequencyDeviations(process, log_frequencies, reference_frequency_index)


class CorrelatedMultiFrequencySky(Model):
    def __init__(
            self,
            prefix: str,
            grid_2d: RegularCartesianGrid,  # FIXME: spatial grid (doesn't have to be 2D)
            log_relative_frequencies: tuple[float],
            zero_mode: Union[tuple, LazyModel],
            zero_mode_offset: float,
            spatial_amplitude: LazyModel,
            spectral_index_mean: Union[tuple, LazyModel],
            spectral_index_fluctuations: Union[tuple, LazyModel],
            spectral_amplitude: LazyModel = None,  # TODO: add option
            spectral_index_deviations: Optional[LazyModel] = None,
            dtype: type = jnp.float64,
    ):
        self.prefix = prefix
        self.hdvol = 1.0 / grid_2d.total_volume
        self.pd = grid_2d.harmonic_grid.power_distributor
        self.ht = partial(hartley, axes=(0, 1))
        self.freqs = jnp.array(log_relative_frequencies)[:, None, None]  # FIXME: spatial grid doesn't have to be 2D
        self.spatial_amplitude = spatial_amplitude
        self.zero_mode_offset = zero_mode_offset
        self._zm = _build_distribution_or_default(
            zero_mode,
            f'{prefix}_zero_mode',
            normal_prior
        )
        self.spectral_index_mean = _build_distribution_or_default(
            spectral_index_mean,
            f'{prefix}_spectral_index_mean',
            normal_prior
        )
        self.spectral_index_fluctuations = _build_distribution_or_default(
            spectral_index_fluctuations,
            f'{prefix}_spectral_index_fluctuations',
            lognormal_prior
        )
        self.spectral_index_deviations = spectral_index_deviations

        models = [self._zm,
                  self.spatial_amplitude,
                  self.spectral_index_mean,
                  self.spectral_index_fluctuations,
                  self.spectral_index_deviations]

        domain = reduce(
            lambda a, b: a | b, [m.domain for m in models if m is not None]
        )

        domain[f"{self.prefix}_spatial_xi"] = ShapeWithDtype(
            grid_2d.shape, dtype)
        domain[f"{self.prefix}_spectral_index_xi"] = ShapeWithDtype(
            grid_2d.shape, dtype)

        super().__init__(self._build_apply(), domain=domain)

    def _build_apply(self):
        if self.spectral_index_deviations is not None:

            def apply_with_deviations(p):
                zm = self.zero_mode(p)

                amplitude = self.spatial_amplitude(p)
                amplitude = amplitude.at[0].set(0.0)
                spatial_xi = p[f"{self.prefix}_spatial_xi"]
                spec_idx_xis = p[f"{self.prefix}_spectral_index_xi"]

                spectral_index = self.spectral_index_fluctuations(p) * spec_idx_xis
                spec_idx_mean = self.spectral_index_mean(p)
                deviations = self.spectral_index_deviations(p)
                distributed_amplitude = amplitude[self.pd]

                terms = (spectral_index * self.freqs + spatial_xi + deviations)
                ht_values = vmap(self.ht)(distributed_amplitude * terms)
                return jnp.exp(self.hdvol * ht_values + spec_idx_mean * self.freqs + zm)

            return apply_with_deviations

        def apply_without_deviations(p):
            # TODO: write a test that checks this vs. apply_with_deviations
            zm = self.zero_mode(p)

            amplitude = self.spatial_amplitude(p)
            amplitude = amplitude.at[0].set(0.0)
            spatial_xi = p[f"{self.prefix}_spatial_xi"]
            spec_idx_xis = p[f"{self.prefix}_spectral_index_xi"]
            spectral_index = self.spectral_index_fluctuations(p) * spec_idx_xis
            spec_idx_mean = self.spectral_index_mean(p)

            spatial_offset = self.hdvol*self.ht(amplitude[self.pd]*spatial_xi)
            spectral_index_spatial = (self.hdvol*self.ht(amplitude[self.pd]*spectral_index)
                                      + spec_idx_mean)
            return jnp.exp(spatial_offset + zm + spectral_index_spatial*self.freqs)

        return apply_without_deviations

    def spectral_index(self, p):
        """Convenience function to retrieve the model's spectral index."""
        amplitude = self.spatial_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        spec_idx_fluctuations = self.spectral_index_fluctuations(p)
        spec_idx_xis = p[f"{self.prefix}_spectral_index_xi"]
        return (self.hdvol*self.ht(amplitude[self.pd]*spec_idx_fluctuations*spec_idx_xis)
                + self.spectral_index_mean(p))

    def spatial_distribution(self, p):
        """Convenience function to retrieve the model's spatial distribution."""
        amplitude = self.spatial_amplitude(p)
        amplitude = amplitude.at[0].set(0.0)
        spatial_xi = p[f"{self.prefix}_spatial_xi"]
        return self.hdvol*self.ht(amplitude[self.pd]*spatial_xi)

    def zero_mode(self, p):
        """Convenience function to retrieve the model's zero mode."""
        return self._zm(p) + self.zero_mode_offset


def build_default_mf_model(
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
    '''
    Build multi frequency model.
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
        - fluctuations: callable or parameters for default (lognormal_prior)

    deviations_settings: If none deviations are not build.
        - process: wiener (default)
            - sigma: callable or parameters for default (lognormal_prior)
    harmonic_type: the type of the harmonic domain
    '''

    grid_2d = _make_grid(shape_2d, distances_2d, harmonic_type)

    zero_mode_model = build_zero_mode_model(prefix, zero_mode_settings)
    spatial_model = build_amplitude_model(prefix, grid_2d, amplitude_settings)
    deviations_model = build_deviations_model(
        prefix, shape_2d, log_frequencies, reference_frequency, deviations_settings)

    return CorrelatedMultiFrequencySky(
        prefix=prefix,
        grid_2d=grid_2d,
        log_relative_frequencies=jnp.array(
            log_frequencies) - log_frequencies[reference_frequency],
        zero_mode=zero_mode_settings['deviations'],
        zero_mode_offset=zero_mode_settings['mean'],
        spatial_amplitude=spatial_model,
        spectral_index_mean=slope_settings['mean'],
        spectral_index_fluctuations=slope_settings['fluctuations'],
        spectral_index_deviations=deviations_model
    )
