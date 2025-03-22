#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Matteo Guardiani

import jax
import jax.random as random
import numpy as np
import pytest

import nifty8 as ift
import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


def _explicify(M, position):
    dim = 0
    for _ in position.values():
        dim += 1
    ravel = lambda x: jax.flatten_util.ravel_pytree(x)[0]
    unravel = lambda x: jax.linear_transpose(ravel, position)(x)[0]
    mat = lambda x: M(unravel(x))
    identity = np.identity(dim, dtype=np.float64)
    m = []
    for v in identity:
        m.append(mat(v))
    return np.vstack(m).T


def get_linear_response(slope_op, intercept_op, sampling_points):
    response = lambda x: slope_op(x) * sampling_points + intercept_op(x)
    return jft.Model(response, domain=slope_op.domain | intercept_op.domain)


@pmp("seed", [0, 42])
def test_estimate_evidence_lower_bound(seed):
    # Set up signal
    n_datapoints = 8

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)

    q = 0.0
    m_slope = 1.5

    sigma_q = 1.5
    sigma_m = 1.8

    intercept = jft.NormalPrior(0.0, sigma_q, name="intercept")
    slope = jft.NormalPrior(0.0, sigma_m, name="slope")

    x = random.uniform(subkey, shape=(n_datapoints,), minval=0.0, maxval=10.0)
    y = q + m_slope * x

    linear_response = get_linear_response(slope, intercept, x)
    key, subkey = random.split(key)
    rp = jft.random_like(subkey, linear_response.domain)

    R = _explicify(linear_response, rp)
    noise_level = 0.8

    noise = random.normal(subkey, shape=(n_datapoints,)) * noise_level
    data = y + noise

    N_inv = np.diag(np.ones_like(data) * (1.0 / noise_level**2))

    S = np.identity(2)
    S_inv = np.identity(2)

    D_inv = R.T @ N_inv @ R + S_inv
    D = np.linalg.inv(D_inv)

    j = R.T @ (N_inv @ data)
    m = D @ j
    m_dag_j = np.dot(m, j)

    det_2pi_D = np.linalg.det(2 * np.pi * D)
    det_2pi_S = np.linalg.det(2 * np.pi * S)

    H_0 = 0.5 * (
        np.vdot(data, (N_inv @ data))
        + n_datapoints * np.log(2 * np.pi * noise_level**2)
        + np.log(det_2pi_S)
        - m_dag_j
    )

    evidence = -H_0 + 0.5 * np.log(det_2pi_D)
    nifty_adjusted_evidence = evidence + 0.5 * n_datapoints * np.log(
        2 * np.pi * noise_level**2
    )
    likelihood_energy = jft.Gaussian(
        data=data, noise_cov_inv=lambda x: (1.0 / noise_level**2) * x
    ).amend(linear_response)

    # Minimization parameters
    n_iterations = 4
    n_samples = 10
    delta = 1e-3
    absdelta = 1e-4

    # Minimize
    samples, _ = jft.optimize_kl(
        likelihood_energy,
        jft.Vector(rp),
        n_total_iterations=n_iterations,
        n_samples=n_samples,
        # Source for the stochasticity for sampling
        key=key,
        draw_linear_kwargs=dict(
            cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10.0, maxiter=100)
        ),
        nonlinearly_update_kwargs=dict(
            minimize_kwargs=dict(
                name="SN",
                xtol=delta,
                cg_kwargs=dict(name=None),
                maxiter=5,
            )
        ),
        kl_kwargs=dict(
            minimize_kwargs=dict(
                name="M", absdelta=absdelta, cg_kwargs=dict(name="MCG"), maxiter=35
            )
        ),
        sample_mode="nonlinear_resample",
        resume=False,
    )

    # Estimate the ELBO
    elbo, stats = jft.estimate_evidence_lower_bound(likelihood_energy, samples, 2)

    assert stats["elbo_lw"] <= nifty_adjusted_evidence <= stats["elbo_up"]


@pmp("seed", [42, 18])
def test_estimate_elbo_nifty_re_vs_nifty(seed):
    # Setup
    key = random.PRNGKey(seed)
    shape = (4, 4)

    cf_zm = {"offset_mean": 0.0, "offset_std": (1e-3, 1e-4)}
    cf_fl = {
        "fluctuations": (1e-1, 5e-3),
        "loglogavgslope": (-3.0, 1e-2),
        "flexibility": None,
        "asperity": None,
    }

    # Test
    cfm = jft.CorrelatedFieldMaker("jcf_")
    cfm.set_amplitude_total_offset(**cf_zm)
    cfm.add_fluctuations(
        shape,
        distances=1.0 / shape[0],
        **cf_fl,
        prefix="",
        non_parametric_kind="power",
    )

    jcf = cfm.finalize()

    n_cfm = ift.CorrelatedFieldMaker("cf_")
    n_cfm.set_amplitude_total_offset(**cf_zm)
    n_cfm.add_fluctuations(ift.RGSpace(shape, 1.0 / shape[0]), **cf_fl)
    cf = n_cfm.finalize(prior_info=0)

    key, subkey = random.split(key)
    pos = jft.random_like(subkey, jcf.domain)

    noise_level = 0.2
    data = jcf(pos) + np.random.normal(0, 1, shape) * noise_level

    like = jft.Gaussian(data, lambda x: 1 / noise_level**2 * x).amend(jcf)

    key, subkey = random.split(key)
    rp = jft.random_like(subkey, jcf.domain)

    n_iterations = 2
    n_samples = 2
    delta = 1e-2
    absdelta = 1e-3

    samples, _ = jft.optimize_kl(
        like,
        jft.Vector(rp),
        n_total_iterations=n_iterations,
        n_samples=n_samples,
        # Source for the stochasticity for sampling
        key=key,
        draw_linear_kwargs=dict(
            cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10.0, maxiter=100)
        ),
        nonlinearly_update_kwargs=dict(
            minimize_kwargs=dict(
                name="SN",
                xtol=delta,
                cg_kwargs=dict(name=None),
                maxiter=5,
            )
        ),
        kl_kwargs=dict(
            minimize_kwargs=dict(
                name="M", absdelta=absdelta, cg_kwargs=dict(name="MCG"), maxiter=35
            )
        ),
        sample_mode="nonlinear_resample",
        resume=False,
    )

    n_samples_list = []
    n_samples = []
    neg = []
    for sample in samples:
        n_samples.append(
            ift.MultiField.from_dict(
                {
                    k[1:]: ift.makeField(cf.domain[k[1:]], v - samples.pos[k])
                    for k, v in sample.tree.items()
                },
                cf.domain,
            )
        )
        neg.append(False)

    n_pos = {
        k[1:]: ift.makeField(cf.domain[k[1:]], v) if k != "cf_spectrum" else v.T
        for k, v in samples.pos.tree.items()
    }
    n_pos = ift.MultiField.from_dict(n_pos, cf.domain)
    n_samples = ift.ResidualSampleList(n_pos, n_samples, neg)

    for samp in n_samples.iterator():
        n_samples_list.append(samp.val)

    N_inv = ift.ScalingOperator(cf.target, 1 / noise_level**2)
    n_like = ift.GaussianEnergy(ift.makeField(cf.target, data), N_inv) @ cf

    n_ham = ift.StandardHamiltonian(n_like)

    elbo, stats = jft.estimate_evidence_lower_bound(like, samples, 4, batch_size=2)
    n_elbo, nstats = ift.estimate_evidence_lower_bound(
        n_ham, n_samples, 4, batch_number=2
    )

    n_elbo_samples = []
    for n_elbo_sample in n_elbo.iterator():
        n_elbo_samples.append(n_elbo_sample.val)
    n_elbo_samples = np.array(n_elbo_samples)

    assert np.allclose(elbo, n_elbo_samples, atol=1e-8)
