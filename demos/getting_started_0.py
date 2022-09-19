# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
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
# ## Wiener filter on two-dimensional field

# +
N_pixels = 256      # Number of pixels
sigma2 = 2.         # Noise variance

def pow_spec(k):
    P0, k0, gamma = [.2, 2, 4]
    return P0 * (1. + (k/k0)**2)**(-gamma/2)

s_space = ift.RGSpace([N_pixels, N_pixels])

# + slideshow={"slide_type": "skip"}
h_space = s_space.get_default_codomain()
HT = ift.HarmonicTransformOperator(h_space,s_space)

# Operators
Sh = ift.create_power_operator(h_space, power_spectrum=pow_spec, sampling_dtype=float)
N = ift.ScalingOperator(s_space, sigma2, sampling_dtype=float)

# Fields and data
sh = Sh.draw_sample()
n = ift.Field.from_random(domain=s_space, random_type='normal',
                      std=np.sqrt(sigma2), mean=0)

# Lose some data

l = int(N_pixels * 0.33)
h = int(N_pixels * 0.33 * 2)

mask = np.full(s_space.shape, 1.)
mask[l:h,l:h] = 0.
mask = ift.Field.from_raw(s_space, mask)

R = ift.DiagonalOperator(mask)(HT)
n = n.val_rw()
n[l:h, l:h] = 0
n = ift.Field.from_raw(s_space, n)
curv = Curvature(R=R, N=N, Sh=Sh)
D = curv.inverse

d = R(sh) + n
j = R.adjoint(N.inverse(d))

# Run Wiener filter
m = D(j)

# Uncertainty
m_mean, m_var = ift.probe_with_posterior_samples(curv, HT, 20, np.float64)

# Get data
s_data = HT(sh).val
m_data = HT(m).val
m_var_data = m_var.val
d_data = d.val
uncertainty = np.sqrt(np.abs(m_var_data))

# + slideshow={"slide_type": "skip"}
cmap = ['magma', 'inferno', 'plasma', 'viridis'][1]

mi = np.min(s_data)
ma = np.max(s_data)

fig, axes = plt.subplots(1, 2)

data = [s_data, d_data]
caption = ["Signal", "Data"]

for ax in axes.flat:
    im = ax.imshow(data.pop(0), interpolation='nearest', cmap=cmap, vmin=mi,
                   vmax=ma)
    ax.set_title(caption.pop(0))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# + slideshow={"slide_type": "skip"}
mi = np.min(s_data)
ma = np.max(s_data)

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
sample = HT(curv.draw_sample(from_inverse=True)+m).val
post_mean = (m_mean + HT(m)).val

data = [s_data, m_data, post_mean, sample, s_data - m_data, uncertainty]
caption = ["Signal", "Reconstruction", "Posterior mean", "Sample", "Residuals", "Uncertainty Map"]

for ax in axes.flat:
    im = ax.imshow(data.pop(0), interpolation='nearest', cmap=cmap, vmin=mi, vmax=ma)
    ax.set_title(caption.pop(0))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Is the uncertainty map reliable?

# + slideshow={"slide_type": "-"}
precise = (np.abs(s_data-m_data) < uncertainty)
print("Error within uncertainty map bounds: " + str(np.sum(precise) * 100 / N_pixels**2) + "%")

plt.imshow(precise.astype(float), cmap="brg")
plt.colorbar()
