# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Jakob Roth

import jax
import jax.numpy as jnp

from .likelihood import Likelihood
from .tree_math import vdot
from .evi import Samples
from .minisanity import minisanity, _rpprint
from .logger import logger


def get_sample_size_estimate(samples):
    import blackjax

    def sample_size_of_key(smpls):
        smpl_size = blackjax.diagnostics.effective_sample_size(smpls[jnp.newaxis,])
        return int(jnp.min(smpl_size))

    samples_sizes = jax.tree.map(sample_size_of_key, samples.samples)
    return samples_sizes


def get_status_message(
    samples,
    state,
    residual=None,
    estimate_effective_sample_size=True,
    *,
    name="",
    map="lmap",
) -> str:
    energy = jnp.mean(-1 * state.logdensity)
    mini_res = ""
    if residual is not None:
        _, mini_res = minisanity(samples, residual, map=map)
    _, mini_pr = minisanity(samples, map=map)
    sample_size = ""
    if estimate_effective_sample_size:
        sample_size = _rpprint(get_sample_size_estimate(samples))
    msg = (
        f"{name}: Mean Energy:{energy:+2.4e}"
        f"\n{name}: Likelihood residual(s):\n{mini_res}"
        f"\n{name}: Prior residual(s):\n{mini_pr}"
        f"\n{name}: Effective samples size(s):\n{sample_size}"
        f"\n"
    )
    return msg


def blackjax_nuts(
    likelihood: Likelihood,
    position,
    key,
    n_warmup_steps=1_000,
    n_samples=1_000,
    estimate_effective_sample_size=True,
    state_and_parameters=None,
):
    """
    Run BlackJAX's NUTS sampler with optional windowed adaptation for step size
    and mass matrix, unless these are provided via ``state_and_parameters``.

    Parameters
    ----------
    likelihood : Likelihood
        Negative log-likelihood of the model.
    position : PyTree
        Initial position in parameter space.
    key : jax.random.PRNGKey
        JAX random key for warm-up and sampling.
    n_warmup_steps : int, default=1000
        Number of warm-up (adaptation) steps used to tune the NUTS sampler.
        Ignored if ``state_and_parameters`` is provided.
    n_samples : int, default=1000
        Number of MCMC samples to draw after warm-up.
    estimate_effective_sample_size : bool, default=True
        Whether to estimate and report the effective sample size in the
        status message.
    state_and_parameters : tuple, optional
        A tuple ``(state, parameters)`` returned from a previous call.
        If provided, warm-up is skipped and sampling resumes from this state
        using the given NUTS parameters. When resuming you need to provide a
        fresh random key, otherwise the same random numbers will be used.

    Returns
    -------
    samples : Samples
        Container holding the drawn HMC samples.
    state_and_parameters : tuple
        A tuple ``(final_state, parameters)`` that can be passed back into
        this function to resume sampling without re-running warm-up.
    """
    import blackjax

    def logdensity(pos):
        return -1.0 * likelihood(pos) - 0.5 * vdot(pos, pos)

    warmup_key, sample_key = jax.random.split(key)
    if state_and_parameters is None:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity,
            adaptation_info_fn=blackjax.adaptation.base.get_filter_adapt_info_fn(),
        )
        (state, parameters), _ = warmup.run(
            warmup_key, position, num_steps=n_warmup_steps
        )
    else:
        state, parameters = state_and_parameters

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    kernel = blackjax.nuts(logdensity, **parameters).step
    kernel = jax.jit(kernel)
    states = inference_loop(sample_key, kernel, state, n_samples)

    samples = Samples(samples=states.position)
    msg = get_status_message(
        samples,
        states,
        likelihood.normalized_residual,
        estimate_effective_sample_size,
        name="BLACKJAX_MCMC",
    )
    logger.info(msg)

    final_state = jax.tree.map(lambda x: x[-1], states)
    return samples, (final_state, parameters)
