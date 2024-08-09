from functools import reduce, partial
from typing import Optional, Union

import jax.numpy as jnp
from jax import vmap
from numpy.typing import ArrayLike

from .. import ShapeWithDtype
from ..num.stats_distributions import lognormal_prior, normal_prior
from ..model import Model
from ..gauss_markov import build_wiener_process
from ..correlated_field import (
    _make_grid,
    hartley,
    RegularCartesianGrid,
    matern_amplitude,
    non_parametric_amplitude,
    WrappedCall)

from .frequency_deviations import FrequencyDeviations


def _add_prefix_to_keys(tree, prefix):
    """Recursively add a prefix to all keys in a nested dictionary."""
    if isinstance(tree, dict):
        return {f"{prefix}_{key}": _add_prefix_to_keys(value, prefix) for key, value in
                tree.items()}
    else:
        return tree


def _remove_prefix_from_keys(tree, prefix):
    """Recursively remove a prefix from all keys in a nested dictionary."""
    if isinstance(tree, dict):
        prefix_length = len(prefix) + 1  # includes the underscore
        return {key[prefix_length:]: _remove_prefix_from_keys(value, prefix) for key, value in
                tree.items() if key.startswith(prefix)}
    else:
        return tree


def _acquire_submodel(model: Optional[Union[Model, None]], prefix: str):
    """
    Acquire a submodel with prefixed domain keys.

    Parameters
    ----------
        model: Model, optional
            The original model.
        prefix: str
            The prefix to add to the domain keys.

    Returns:
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


def _check_demands(model_name, kwargs, demands):
    for key in demands:
        assert key in kwargs, (f'{key} not in {model_name}.\n'
                               f'Provide settings for {key}')


def _set_default_or_call(arg: Union[callable, tuple, list],
                         default: callable):
    """Either sets the default distribution or the callable"""
    if callable(arg):
        # TODO: do a check here that it is a valid distribution.
        return arg
    return default(*arg)


def _safe_set_default_or_call(arg: Union[callable, tuple, list, None],
                              default: callable):
    """Either sets the default distribution or the callable"""
    return _set_default_or_call(arg, default) if arg is not None else None


def _build_distribution_or_default(arg: Union[callable, tuple, list],
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
    grid: RegularCartesianGrid,
    settings: dict,
    amplitude_model: str = "non_parametric",
    renormalize_amplitude: bool = False,
    prefix: str = None,
    kind: str = "amplitude",
) -> Model:
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


def build_deviations_model(
    shape: tuple[int],
    log_frequencies: Union[tuple[float], ArrayLike],
    reference_frequency_index: int,
    deviations_settings: Optional[dict],
    prefix: str = None,
) -> FrequencyDeviations | None:

    if deviations_settings is None:
        return None

    dev_name = f'{prefix}_deviations' if prefix is not None else 'deviations'
    process_name = deviations_settings.get('process', 'wiener').lower()

    # FIXME: Need more processes
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


class CorrelatedMultiFrequencySky(Model):
    def __init__(
        self,
        prefix: str,
        grid: RegularCartesianGrid,
        log_relative_frequencies: Union[tuple[float], ArrayLike],
        zero_mode: Model,
        zero_mode_offset: float,
        spatial_amplitude: Model,
        spectral_index_mean: Model,
        spectral_index_fluctuations: Model,
        spectral_amplitude: Model = None,  # TODO: add option
        spectral_index_deviations: Optional[Model] = None,
        dtype: type = jnp.float64,
    ):
        self.prefix = prefix
        slicing_tuple = (slice(None),) + (None,) * len(grid.shape)
        self._freqs = jnp.array(log_relative_frequencies)[slicing_tuple]
        self.hdvol = 1.0 / grid.total_volume
        self.pd = grid.harmonic_grid.power_distributor
        self.ht = partial(hartley, axes=tuple(range(len(grid.shape))))
        self.zero_mode_offset = zero_mode_offset
        self._zm = _acquire_submodel(
            zero_mode, prefix)
        self.spatial_amplitude = _acquire_submodel(
            spatial_amplitude, prefix)
        self.spectral_index_mean = _acquire_submodel(
            spectral_index_mean, prefix)
        self.spectral_index_fluctuations = _acquire_submodel(
            spectral_index_fluctuations, prefix)
        self.spectral_index_deviations = _acquire_submodel(
            spectral_index_deviations, prefix)

        models = [self._zm,
                  self.spatial_amplitude,
                  self.spectral_index_mean,
                  self.spectral_index_fluctuations,
                  self.spectral_index_deviations]

        domain = reduce(
            lambda a, b: a | b, [m.domain for m in models if m is not None]
        )

        domain[f"{self.prefix}_spatial_xi"] = ShapeWithDtype(
            grid.shape, dtype)
        domain[f"{self.prefix}_spectral_index_xi"] = ShapeWithDtype(
            grid.shape, dtype)

        super().__init__(self._build_apply(), domain=domain)

    def _build_apply(self):
        if self.spectral_index_deviations is not None:

            def apply_with_deviations(p):
                zm = self.zero_mode(p)

                amplitude = self.spatial_amplitude(p)
                amplitude = amplitude.at[0].set(0.0)
                spatial_xi = p[f"{self.prefix}_spatial_xi"]
                spec_idx_xis = p[f"{self.prefix}_spectral_index_xi"]

                spectral_index = self.spectral_index_fluctuations(
                    p) * spec_idx_xis
                spec_idx_mean = self.spectral_index_mean(p)
                deviations = self.spectral_index_deviations(p)
                distributed_amplitude = amplitude[self.pd]

                terms = (spectral_index*self._freqs + spatial_xi + deviations)
                ht_values = vmap(self.ht)(distributed_amplitude * terms)
                return jnp.exp(self.hdvol * ht_values + spec_idx_mean * self._freqs + zm)

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
            return jnp.exp(spatial_offset + zm + spectral_index_spatial * self._freqs)

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
    shape: tuple[int],
    distances: tuple[float],
    log_frequencies: Union[tuple[float], ArrayLike],
    reference_frequency_index: int,
    zero_mode_settings: dict,
    amplitude_settings: dict,
    spectral_index_settings: dict,
    deviations_settings: Optional[dict] = None,
    amplitude_model: str = "non_parametric",
    harmonic_type: str = 'fourier',
):
    """
    Build multi-frequency sky model.
        f = exp( F * A * (
              io(k, l0) +
              slope(k) * (l-l0) +
              GaussMarkovProcess(k, l-l0) (Nd)
              - MeanSlope(GMP) (N-1d)
        ) * (offset_mean + deviations) )

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

    amplitude_settings: dict
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

    deviations_settings: dict, opt
        Settings for the spectral index priors.
        If none deviations are not build.
        Should contain the following keys:
        - process: wiener (default)
        - sigma: callable or parameters
             for default (lognormal prior)

    amplitude_model: str, optional
        Amplitude model to be used.
        By default, the correlated field model
        (`'non_parametric'`).

    harmonic_type: str, optional
        The type of the harmonic domain for the amplitude model.
    """

    grid = _make_grid(shape, distances, harmonic_type)

    spatial_model = build_amplitude_model(grid, amplitude_settings,
                                          amplitude_model=amplitude_model)
    deviations_model = build_deviations_model(shape,
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
        log_relative_frequencies=jnp.array(
            log_frequencies) - log_frequencies[reference_frequency_index],
        zero_mode=zero_mode,
        zero_mode_offset=zero_mode_settings['mean'],
        spatial_amplitude=spatial_model,
        spectral_index_mean=spectral_index_mean,
        spectral_index_fluctuations=spectral_index_fluctuations,
        spectral_index_deviations=deviations_model
    )
