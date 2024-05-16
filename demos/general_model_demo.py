# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Copyright(C) 2024
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Margret Westerkamp, Vincent Eberle

from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

import nifty8.re as jft

seed = 42
key = random.PRNGKey(seed)
# -

# # A General Multifrequency Model
#
# $I(x, \nu, \nu_0) = \exp(\alpha \cdot \log (\nu/\nu_0) ) \times I_{0}(x)$
#
# where $ I_{0}(x) = I_{\text{spatial}}(x) $
#
# What we basically implement is:
#
# $\log I(x, \nu, \nu_0) = \alpha \cdot \log(\nu/\nu_0) + \log I_{spatial}(x)$
#
# Here
# $\alpha = \alpha(x)$
#
# and also you can add a term for deviations from the powerlaw, $\delta$:
#
# $\log I(x, \nu, \nu_0) = \alpha \times \log(\nu/\nu_0) + \log I_{spatial}(x) + \delta (x, \Delta(\nu/\nu_0))$
#
# This $\delta$ has the same correlation structure along $\nu$ but can have different $\xi$'s for every $x$
#

# # Fields

# +
e_dims = (12)
s_dims = (512,512)

RG_Energies = False
if RG_Energies:
    freqs = jnp.arange(0, 12)
else:
    freqs = jnp.array([1, 3, 4, 7, 12, 17, 19.3, 22, 25, 25.1, 25.3, 26]) # Should be log nu
    dfreqs = freqs[1:]-freqs[:-1]
# -

# ## Spatial field $I_0$

cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
cf_fl = {
    "fluctuations": (1.0, 5e-3),
    "loglogavgslope": (-3., 1e-2),
    "flexibility": (1e+0, 5e-1),
    "asperity": None,
    # "harmonic_type": "Fourier"
}
cfm = jft.CorrelatedFieldMaker("space_cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    s_dims,
    distances=1. / s_dims[0],
    **cf_fl,
    prefix="ax1",
    non_parametric_kind="power"
)
correlated_field = cfm.finalize()

# ## Spectral Index $\alpha$

alpha_zm = {"offset_mean": -2.0, "offset_std": (1e-3, 1e-4)}
alpha_fl = {
    "fluctuations": (1, 5e-3),
    "loglogavgslope": (-4., 1e-2),
    "flexibility": (1e-1, 5e-2),
    "asperity": None,
    #"harmonic_type": "Fourier"
}
alpha = jft.CorrelatedFieldMaker("alpha")
alpha.set_amplitude_total_offset(**alpha_zm)
alpha.add_fluctuations(
    s_dims,
    distances=1. / s_dims[0],
    **alpha_fl,
    prefix="ax1",
    non_parametric_kind="power"
)
alpha_field = alpha.finalize()

# # Build Power Law

plaw = jft.build_power_law(freqs, alpha_field)

# ## Deviations from Powerlaw

if RG_Energies:
    dev_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
    dev_fl = {
        "fluctuations": (1e-1, 5e-3),
        "loglogavgslope": (-1., 1e-2),
        "flexibility": (1e-3, 5e-1),
        "asperity": None,
    }
    dev_cfm = jft.CorrelatedFieldMaker("dev_cf")
    dev_cfm.set_amplitude_total_offset(**dev_zm)
    dev_cfm.add_fluctuations(
        e_dims,
        distances=1. / e_dims,
        **dev_fl,
        prefix="ax1",
        non_parametric_kind="power"
    )
    dev_cf = dev_cfm.finalize()
    dev = jft.MappedModel(dev_cf, "dev_cfxi", s_dims, False)
else:
    wp = jft.WienerProcess((0), (1, 3), dfreqs, name="mapped_wp", N_steps=e_dims-1)
    dev = jft.MappedModel(wp, "mapped_wp", s_dims, False)

# # General Multifrequency Model

gen_mod = jft.GeneralModel({'spatial': correlated_field, 'freq_plaw': plaw, 'freq_dev':dev}).build_model()

key, subkey = random.split(key)
pos_init = jft.Vector(jft.random_like(subkey, gen_mod.domain))
result = gen_mod(pos_init)

# # Plot $I(x, \nu)$ for some $\nu$ (Energy-slices)

for j in range(12):
    plt.imshow(result[j,:,:])
    plt.colorbar()
    plt.show()
    plt.close()

# # Plot  Plot $I(x, \nu)$ for some $x$ (spatial-slices) 

for i in range(20):
    plt.plot(freqs, result[:, i*5, i*5])
plt.show()
plt.close()

# # Plot $I_0$

plt.imshow(correlated_field(pos_init))
plt.colorbar()

# # Plot $\alpha$

plt.imshow(alpha_field(pos_init))
plt.colorbar()


