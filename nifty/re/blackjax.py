# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Jakob Roth

import blackjax
import jax
import jax.numpy as jnp

from .likelihood import Likelihood
from .tree_math import vdot, zeros_like
from .evi import Samples
from .minisanity import minisanity, _rpprint
from .logger import logger


def get_sample_size_estimate(samples):
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
):

    def logdensity(pos):
        return -1.0 * likelihood(pos) - 0.5 * vdot(pos, pos)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        adaptation_info_fn=blackjax.adaptation.base.get_filter_adapt_info_fn(),
    )
    warmup_key, sample_key = jax.random.split(key)
    (state, parameters), _ = warmup.run(warmup_key, position, num_steps=n_warmup_steps)

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

    hmc_samples = states.position

    samples = Samples(pos=zeros_like(likelihood.domain), samples=hmc_samples)
    msg = get_status_message(
        samples,
        states,
        likelihood.normalized_residual,
        estimate_effective_sample_size,
        name="BLACKJAX_MCMC",
    )
    logger.info(msg)

    return samples, states
