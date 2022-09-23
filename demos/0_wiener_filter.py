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
# ## Introduction to Information Field Theory(IFT)
# Typically we start with the measurement equation
# $$d_i = (Rs)_i+n_i$$
# Here, $s$ is a continuous field, $d$ a discrete data vector, $n$ is the discrete noise on each data point and $R$ is the instrument response. In most cases, $R$ is not invertible. IFT aims at **inverting** the above uninvertible problem in the **best possible way** using Bayesian statistics.
#
# NIFTy (Numerical Information Field Theory) is a Python framework in which IFT problems can be tackled easily.
#
# Its main interfaces are:
#
# - **Spaces**: Cartesian, 2-Spheres (Healpix, Gauss-Legendre) and their respective harmonic spaces.
# - **Fields**: Defined on spaces.
# - **Operators**: Acting on fields.

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Wiener filter on 1D- fields in IFT
#
# ### Assumptions
# - We consider a linear response R in the measurement equation $d=Rs+n$.
# - We also assume the signal and the noise prior to be Gaussian  $\mathcal P (s) = \mathcal G (s,S)$, $\mathcal P (n) = \mathcal G (n,N)$ 
# - Here $S, N$ are signal and noise covariances. Therefore they are positive definite matrices.
#
# ### Wiener filter solution
# - Making use of Bayes' theorem, the posterior is proportional to the joint probability and is given by:
#
# $$\mathcal P (s|d) \propto P(s,d) = \mathcal G(d-Rs,N) \,\mathcal G(s,S) \propto \mathcal G (s-m,D)$$
#
# - Here, m is the posterior mean , D is the information propagator.
# - $m = Dj, D = (S^{-1} +R^\dagger N^{-1} R)^{-1} , \quad j = R^\dagger N^{-1} d.$
#
# Let us implement this in NIFTy!

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Implementation in NIFTy
#
# - We assume statistical homogeneity and isotropy. Therefore the signal covariance $S$ is diagonal in harmonic space, and is described by a one-dimensional power spectrum, assumed here to be a power-law, $$P(k) = P_0\,\left(1+\left(\frac{k}{k_0}\right)^2\right)^{-\gamma /2},$$ with $P_0 = 20000, k_0 = 5, \gamma = 4$, thus the reconstruction starts in harmonic space. 
# - We define a Cartesian space with $N_{pix} = 512$ being the number of grid cells.
# - We assume the noise covariance to be uncorrelated and constant, $N = 0.2 \cdot \mathbb{1}$.
# - For simplicity we define the response operator as a unit matrix, $R = \mathbb{1}$.
#

# + slideshow={"slide_type": "-"}
# %matplotlib inline
import numpy as np
import nifty8 as ift
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.style.use("seaborn-notebook")


# + slideshow={"slide_type": "-"}
def pow_spec(k):
    P0, k0, gamma = [2e4, 5, 4]
    return P0 / ((1. + (k/k0)**2)**(gamma / 2))


# -

N_pix = 512
s_space = ift.RGSpace(N_pix)

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

# + slideshow={"slide_type": "-"}
npix = d_space.size
l = int(npix * 0.2)
h = int(npix * 0.2 * 2)

mask = np.zeros(d_space.shape, bool)
mask[l:h] = 1
mask = ift.makeField(d_space, mask)
maskOp = ift.MaskOperator(mask)
R1 = maskOp @ R

# + slideshow={"slide_type": "-"}
N1 = ift.ScalingOperator(R1.target, noise_amplitude**2, float)
n1 = N1.draw_sample()
d1 = R1(s) + n1

# + slideshow={"slide_type": "skip"}
Dinv = ift.SamplingEnabler(ift.SandwichOperator.make(cheese=N1.inverse, bun=R1), S.inverse, ic, S.inverse)
Dinv = ift.InversionEnabler(Dinv, ic)
D1 = Dinv.inverse
j1 = R1.adjoint(N1.inverse(d1))
m1 = D1(j1)
# -

n_samples = 200
sc = ift.StatCalculator()
for _ in range(n_samples):
    sample = m1 + D1.draw_sample()
    sc.add(sample)
m_std = sc.var.sqrt()

# + slideshow={"slide_type": "skip"}
plt.axvspan(l, h, facecolor='0.8',alpha=0.5)
plt.fill_between(range(m1.size), (m1 - m_std).val, (m1 + m_std).val, facecolor='0.5', alpha=0.5, label="Sample std")
plt.plot(sc.mean.val, 'k--', label="Sample mean")
plt.plot(s.val, 'r', label="Signal", alpha=1, linewidth=2)

tmp = maskOp.adjoint(d1).val_rw()
tmp[tmp == 0.] = np.nan
plt.plot(tmp, 'k.', label="Data")

plt.plot(m1.val, 'k', label="Reconstruction", linewidth=2)
plt.title("Reconstruction of incomplete data")
plt.legend()
plt.show()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Wiener Filter standardized
# -
sqrt_pspec = Sh(ift.full(Sh.domain, 1.)).sqrt()
trafo = HT.adjoint @ ift.makeOp(sqrt_pspec)
R2 = R1 @ trafo
j2 = R2.adjoint(N1.inverse(d1))
identity = ift.Operator.identity_operator(R2.domain)
Dinv = ift.InversionEnabler(identity + R2.adjoint @ N1.inverse @ R2, ic)
D2 = Dinv.inverse
m2 = D2(j2)

m2_s_space = trafo(m2)
plt.axvspan(l, h, facecolor='0.8',alpha=0.5)
plt.plot(s.val, 'r', label="Signal", alpha=1, linewidth=2)
plt.plot(tmp, 'k.', label="Data")
plt.plot(m2_s_space.val, 'k', label="Reconstruction", linewidth=2)
plt.title("Reconstruction of incomplete data in normalized coordinates")
plt.legend()
plt.show()
