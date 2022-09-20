# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# # Code example: Wiener filter

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Introduction
# IFT starting point:
#
# $$d = Rs+n$$
#
# Typically, $s$ is a continuous field, $d$ a discrete data vector. Particularly, $R$ is not invertible.
#
# IFT aims at **inverting** the above uninvertible problem in the **best possible way** using Bayesian statistics.
#
# NIFTy (Numerical Information Field Theory) is a Python framework in which IFT problems can be tackled easily.
#
# Main Interfaces:
#
# - **Spaces**: Cartesian, 2-Spheres (Healpix, Gauss-Legendre) and their respective harmonic spaces.
# - **Fields**: Defined on spaces.
# - **Operators**: Acting on fields.

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Wiener filter on one-dimensional fields
#
# ### Assumptions
#
# - $d=Rs+n$, $R$ linear operator.
# - $\mathcal P (s) = \mathcal G (s,S)$, $\mathcal P (n) = \mathcal G (n,N)$ where $S, N$ are positive definite matrices.
#
# ### Posterior
# The Posterior is given by:
#
# $$\mathcal P (s|d) \propto P(s,d) = \mathcal G(d-Rs,N) \,\mathcal G(s,S) \propto \mathcal G (s-m,D) $$
#
# where
# $$m = Dj$$
# with
# $$D = (S^{-1} +R^\dagger N^{-1} R)^{-1} , \quad j = R^\dagger N^{-1} d.$$
#
# Let us implement this in NIFTy!

# + [markdown] slideshow={"slide_type": "subslide"}
# ### In NIFTy
#
# - We assume statistical homogeneity and isotropy. Therefore the signal covariance $S$ is diagonal in harmonic space, and is described by a one-dimensional power spectrum, assumed here as $$P(k) = P_0\,\left(1+\left(\frac{k}{k_0}\right)^2\right)^{-\gamma /2},$$
# with $P_0 = 20000, k_0 = 5, \gamma = 4$.
# - $N = 0.2 \cdot \mathbb{1}$.
# - Number of data points $N_{pix} = 512$.
# - reconstruction in harmonic space.
# - Response operator:
# $$R = \mathbb{1}$$
#

# + slideshow={"slide_type": "-"}
def pow_spec(k):
    P0, k0, gamma = [2e4, 5, 4]
    return P0 / ((1. + (k/k0)**2)**(gamma / 2))

s_space = ift.RGSpace(512)

# + [markdown] slideshow={"slide_type": "slide"}
# ### Implementation

# + slideshow={"slide_type": "-"}
# %matplotlib inline
import numpy as np
import nifty8 as ift
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.style.use("seaborn-notebook")
# -

HT = ift.HartleyOperator(s_space)

Sh = ift.create_power_operator(HT.target, power_spectrum=pow_spec, sampling_dtype=float)

S = HT.adjoint @ Sh @ HT
S = ift.SandwichOperator.make(bun=HT, cheese=Sh)

s = S.draw_sample()

R = ift.GeometryRemover(s_space)
d_space = R.target

# +
noiseless_data = R(s)
noise_amplitude = np.sqrt(0.2)
N = ift.ScalingOperator(d_space, noise_amplitude**2, float)

n = N.draw_sample()
d = noiseless_data + n
j = R.adjoint(N.inverse(d))

ic = ift.GradientNormController(iteration_limit=50000, tol_abs_gradnorm=0.1)
Dinv = ift.InversionEnabler(S.inverse + R.adjoint @ N.inverse @ R, ic)
D = Dinv.inverse

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Run Wiener Filter

# + slideshow={"slide_type": "-"}
m = D(j)

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Results

# + slideshow={"slide_type": "-"}
plt.plot(s.val, 'r', label="Signal", linewidth=2)
plt.plot(d.val, 'k.', label="Data")
plt.plot(m.val, 'k', label="Reconstruction",linewidth=2)
plt.title("Reconstruction")
plt.legend()
plt.show()

# + slideshow={"slide_type": "subslide"}
plt.plot(s.val - s.val, 'r', label="Signal", linewidth=2)
plt.plot(d.val - s.val, 'k.', label="Data")
plt.plot(m.val - s.val, 'k', label="Reconstruction",linewidth=2)
plt.axhspan(-noise_amplitude,noise_amplitude, facecolor='0.9', alpha=.5)
plt.title("Residuals")
plt.legend()
plt.show()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Wiener Filter on Incomplete Data

# + slideshow={"slide_type": "skip"}
# Operators
Sh = ift.create_power_operator(h_space, power_spectrum=pow_spec, sampling_dtype=float)
N = ift.ScalingOperator(s_space, noise_amplitude**2, sampling_dtype=float)
# R is defined below

# Fields
sh = Sh.draw_sample()
s = HT(sh)
n = N.draw_sample()

# + [markdown] slideshow={"slide_type": "skip"}
# ### Partially Lose Data

# + slideshow={"slide_type": "-"}
l = int(N_pixels * 0.2)
h = int(N_pixels * 0.2 * 2)

mask = np.full(s_space.shape, 1.)
mask[l:h] = 0
mask = ift.Field.from_raw(s_space, mask)

R = ift.DiagonalOperator(mask) @ HT
n = n.val_rw()
n[l:h] = 0
n = ift.Field.from_raw(s_space, n)

d = R(sh) + n

# + slideshow={"slide_type": "skip"}
curv = Curvature(R=R, N=N, Sh=Sh)
D = curv.inverse
j = R.adjoint(N.inverse(d))
m = D(j)

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Compute Uncertainty
#
# -

m_mean, m_var = ift.probe_with_posterior_samples(curv, HT, 200, np.float64)

# + [markdown] slideshow={"slide_type": "skip"}
# ### Get data

# + slideshow={"slide_type": "skip"}
# Get signal data and reconstruction data
s_data = s.val
m_data = HT(m).val
m_var_data = m_var.val
uncertainty = np.sqrt(m_var_data)
d_data = d.val_rw()

# Set lost data to NaN for proper plotting
d_data[d_data == 0] = np.nan

# + slideshow={"slide_type": "skip"}
plt.axvspan(l, h, facecolor='0.8',alpha=0.5)
plt.fill_between(range(N_pixels), m_data - uncertainty, m_data + uncertainty, facecolor='0.5', alpha=0.5)
plt.plot(s_data, 'r', label="Signal", alpha=1, linewidth=2)
plt.plot(d_data, 'k.', label="Data")
plt.plot(m_data, 'k', label="Reconstruction", linewidth=2)
plt.title("Reconstruction of incomplete data")
plt.legend()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Wiener Filter standardized
# + [markdown] slideshow={"slide_type": "slide"}

# FIXME
