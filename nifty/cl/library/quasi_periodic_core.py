import numpy as np
import nifty.cl as ift
import matplotlib.pyplot as plt
import jax.numpy as jnp

from quasi_periodic_prior import build_prior_operator, _transform_setting


def build_gaussian_periodicity_model(
    pspace,
    domain_key: str,
    periodicity: float,
    jmax: int,
    amplitude: dict,
    sigma: dict,
    **params: dict,
):
    '''Build a Gaussian quasi-periodic model. The model builds a Gaussian for
    multiple frequencies, according to:


    Params:
    -------
    domain_key: str
        The base key for the domain of the model

    periodicity: float
        The periodicity of the signal in frequency.

    jmax: int
        The maximum multiplicity of the periodicity considered in the model.

    amplitude: dict
        The settings for the prior of the log amplitude model, which needs to
        include priors for the:
           - log_intercept
           - log_slope
           - log_delta

    sigma: dict
        The settings for the prior of the log sigma model.
    '''
    amplitude_key = domain_key + '_amplitude'
    sigma_key = domain_key + '_sigma'
    log_amplitude_prior = build_log_amplitude_prior(amplitude_key, amplitude)
    log_sigma_prior = build_log_sigma_prior(sigma_key, sigma)

    prior = log_sigma_prior + log_amplitude_prior

    frequencies = pspace.k_lengths

    def GaussianOperator(x):
        log_amplitudes, log_sigmas = x[amplitude_key], x[sigma_key]
        out = 0
        for j in range(1, jmax):
            log_a = log_amplitudes[j-1]
            log_s = log_sigmas[j-1]

            out += jnp.exp(log_a)*jnp.exp(
                -0.5 * (frequencies-periodicity*j)**2 /
                jnp.exp(log_s)**2)

        return out

    gop = ift.JaxOperator(prior.target, pspace, GaussianOperator)
    return gop @ prior


def build_log_amplitude_prior(domain_key: str, prior_params: dict, jmax: int = 8):
    li_key = domain_key + '_a0'
    li_vals = _transform_setting(prior_params.get('amplitude_log_intercept'))
    li_vals['N_copies'] = 1

    sl_key = domain_key + '_slope'
    sl_vals = _transform_setting(prior_params.get('amplitude_slope'))
    sl_vals['N_copies'] = 1

    d_key = domain_key + '_delta'
    d_vals = _transform_setting(prior_params.get('amplitude_delta_sigma'))
    d_vals['N_copies'] = jmax - 1
    assert d_vals['mean'] == 0.0

    a0_prior = build_prior_operator(li_key, li_vals)
    slope_prior = build_prior_operator(sl_key, sl_vals)
    delta_prior = build_prior_operator(d_key, d_vals)

    prior = a0_prior + slope_prior + delta_prior

    alpha_cast = jnp.arange(jmax)

    def log_amplitude_func(x):
        a0, alpha = x[li_key], x[sl_key]
        deltas = jnp.zeros(jmax)
        deltas = deltas.at[1:].set(x[d_key])
        return a0 - alpha_cast * alpha + deltas

    log_amplitude_model = ift.JaxOperator(
        domain=prior.target,
        target=ift.UnstructuredDomain(jmax),
        func=log_amplitude_func).ducktape_left(domain_key)

    return log_amplitude_model @ prior


def build_log_sigma_prior(domain_key: str, prior_params: dict, jmax: int = 8):
    sigma_key = domain_key
    sigma_vals = _transform_setting(prior_params.get('log_sigma'))
    sigma_vals['N_copies'] = jmax

    sigma_prior = build_prior_operator(sigma_key, sigma_vals)

    return sigma_prior
