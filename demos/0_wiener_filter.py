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
#

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Wiener filter on 1D- fields in IFT
#
# ### Assumptions
# - We consider a linear response R in the measurement equation $d=Rs+n$.
# - We also assume the **signal** and the **noise** prior to be **Gaussian**  $\mathcal P (s) = \mathcal G (s,S)$, $\mathcal P (n) = \mathcal G (n,N)$ 
# - Here $S, N$ are signal and noise covariances. Therefore they are positive definite matrices.
#
# ### Wiener filter solution
# - Making use of Bayes' theorem, the posterior is proportional to the joint probability and is given by:
#
# $$\mathcal P (s|d) \propto P(s,d) = \mathcal G(d-Rs,N) \,\mathcal G(s,S) \propto \mathcal G (s-m,D)$$
#
# - Here, $m$ is the posterior mean , $D$ is the information propagator and are defined as follows:
# $$m = Dj, \quad D = (S^{-1} +R^\dagger N^{-1} R)^{-1} $$
# - There, $j$ is the information source defined as $$ j = R^\dagger N^{-1} d.$$
#
# Let us implement this in **NIFTy!** So let's import all the packages we need. :D
# -

# %matplotlib inline
import numpy as np
import nifty8 as ift
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.style.use("seaborn-notebook")


# + [markdown] slideshow={"slide_type": "subslide"}
# ### Implementation in NIFTy
#
# - We assume statistical **homogeneity** and **isotropy**, so the signal covariance $S$ is **translation invariant** and only depends on the **absolute value** of the distance. According to Wiener-Khinchin theorem, the signal covariance $S$ is diagonal in harmonic space, $$S_{kk^{\prime}} = 2 \pi \delta(k-k^{\prime}) P(k)= \text{diag}(S) \equiv \widehat{S_k}$$
# and is described by a one-dimensional power spectrum. We assume the power spectrum to follow a power-law, $$P(k) = P_0\,\left(1+\left(\frac{k}{k_0}\right)^2\right)^{-\gamma /2},$$ with $P_0 = 2 \cdot 10^4, \ k_0 = 5, \ \gamma = 4$, thus the reconstruction starts in harmonic space. 

# + slideshow={"slide_type": "-"}
def pow_spec(k):
    P0, k0, gamma = [2e4, 5, 4]
    return P0 / ((1. + (k/k0)**2)**(gamma / 2))


# -

# ### Spaces and harmonic transformations
# - We define our non-harmonic signal space to be Cartesian with $N_{pix} = 512$ being the number of grid cells.
# - To connect harmonic and non-harmonic spaces we introduce the Hartley transform $H$ that is closely related to the Fourier transform but maps $\mathbb{R}\rightarrow\mathbb{R}$.
# - The covariance S in non-harmonic space is given by $$S = H^{\dagger}\widehat{S_k} H \ .$$

N_pix = 512
s_space = ift.RGSpace(N_pix) # signal space is a regular Cartesian grid space
HT = ift.HartleyOperator(s_space)
k_space = HT.target # k_space is the harmonic conjugate space of s_space

S_k = ift.create_power_operator(k_space, power_spectrum=pow_spec, sampling_dtype=float)

S = ift.SandwichOperator.make(bun=HT, cheese=S_k) # Sandwich Operator implements S = HT.adjoint @ S_k @ HT and enables NIFTy to sample from S

# ### Synthetic Data
# - In order to demonstrate the Wiener filter, we are using **synthetic data**. Therefore, we draw a sample $\tilde{s}$ from $S$. (see Sampling)
# - For simplicity we define the response operator as a unit matrix, $R = \mathbb{1}$.
# - We assume the noise covariance to be uncorrelated and constant, $N = 0.2 \cdot \mathbb{1}$ and draw a sample $\tilde{n}$.
# - Thus the synthetic data $d = R(\tilde{s}) + \tilde{n}$.

# ### Sampling
#
# - Assuming we have a distribution $\mathcal{G}(b,B)$ we can sample from and we want to draw a sample from a distritbution $\mathcal{G}(c,C)$ with covariance $C$. The two distributions are connected via the relation $C = ABA^{\dagger}.$ One can show that $c= Ab$ with $b \curvearrowleft \mathcal{G}(b,B)$	has a probability distribution with covariance $C$ as desired. 
# $$ \langle cc^\dagger\rangle_{\mathcal{G}(c,C)} = \langle Ab(Ab)^\dagger\rangle_{\mathcal{G}(b,B)} = \langle Abb^\dagger A^\dagger \rangle =  A \langle bb^\dagger  \rangle A^\dagger = ABA^\dagger = C$$
# - This is also true for the case that $B = \mathbb{1}$, meaning that $\mathcal{G}(b,\mathbb{1})$ Thus $C = AA^{\dagger}$ .
# - Note that, if $C$ is diagonal, $A$ is diagonal as well.
# - It can be shown that if $C = A + B$, then $c = a + b$ with $b \curvearrowleft \mathcal{G}(b,B)$ and $a \curvearrowleft \mathcal{G}(a,A)$ has a probability distribution with covariance $C$ as desired. 
# - If we can draw samples from $\mathcal{G}(a,A)$, and we want to draw a sample with the covariance $A^{-1}$, one can simply show that $c = A^{-1}a$ has a  a probability distribution with covariance $A^{-1}$.
# $$\langle c c^{\dagger} \rangle = \langle A^{-1}aa^{\dagger}(A^{-1})^{\dagger} \rangle =  A^{-1}\langle aa^{\dagger}\rangle(A^{-1})^{\dagger} = A^{-1} A(A^{-1})^{\dagger} =A^{-1}$$
# as we assume $A^{-1}$ to be Hermitian.
#
# By this brief introduction to sampling, we apply it in order to get the synthetic data. All of these sampling rules are implemented in NIFTy so we do not need to take care. 

s = S.draw_sample() # Drawing a sample from signal with covariance S.

R = ift.GeometryRemover(s_space) # Defining the responce operator that removes the geometry meaning it removes distances and volumes.
d_space = R.target # Defining the data space that has an unstructured domain.

# +
noiseless_data = R(s)

noise_amplitude = np.sqrt(0.2) # this is the multiplicative factor going from a sample with unit covariance to N
N = ift.ScalingOperator(d_space, noise_amplitude**2, float) # Defining the noise covariance.
n = N.draw_sample() #Drawing a sample from noise with covariance N.

d = noiseless_data + n  # Synthetic data!

# -

# ### Information Source and Information Propagator
#
#  Now that we have the synthetic data, we are one step closer to the Wiener filter! In order to apply Wiener filter on the data we first need to define the information source $j$ along with the information propagator $D$.   

# +
j = R.adjoint(N.inverse(d)) # Defining the information propagator.

ic = ift.GradientNormController(iteration_limit=50000, tol_abs_gradnorm=0.1) # Iteration controller
D_inv = S.inverse + R.adjoint @ N.inverse @ R
D_inv = ift.InversionEnabler(D_inv, ic) # Enabling .inverse to invert D via Conjugate Gradient.
D = D_inv.inverse

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Applying Wiener Filter 
#
# After defining the information source and propagator, we are able to apply the Wiener filter in order to get the posterior mean $m = \langle s \rangle_{\mathcal{P}(s|d)}$ that is our reconstruction of the signal:

# + slideshow={"slide_type": "-"}
m = D(j)

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Results

# + slideshow={"slide_type": "-"}
plt.plot(s.val, 'r', label="signal ground truth", linewidth=2) # .val retrieves numpy array from fields.
plt.plot(d.val, 'k.', label="noisy data")
plt.plot(m.val, 'k', label="posterior mean",linewidth=2)
plt.title("Reconstruction")
plt.legend()
plt.show()
# -

# In order to show the deviations with respect to the true signal (ground truth), we  plot the residuals as follows:

# + slideshow={"slide_type": "subslide"}
plt.plot(s.val - s.val, 'r', label="ground truth ref [$s-s$]", linewidth=2)
plt.plot(d.val - s.val, 'k.', label="noise [$d-Rs$]")
plt.plot(m.val - s.val, 'k', label="posterior mean - ground truth",linewidth=2)
plt.axhspan(-noise_amplitude,noise_amplitude, facecolor='0.9', alpha=.5)
plt.title("Residuals")
plt.legend()
plt.show()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Wiener Filter on Incomplete Data
#
# Now we consider a case that the data is not complete. This might be the case in real situations as the instrument might not be able to receive data. In order to apply the Wiener filter to this case, we first need to build the response corresponding to the incomplete measurement in NIFTy!
# -

# ### Incomplete Measuring / Masking 
# We need to build mask operator which cuts out all the unobserved parts of the signal. Lets assume that we first observe the signal for some time, but then something goes wrong with our instrument and we don't collect data for a while. After fixing the instrument we can collect data again. This means that data lives on an unstructured domain as the there is data missing for the period of time $t_{\text{off}}$ when the instrument was offline. In order to implement this incomplete measurement we need to define a new response operator $R$ which masks the signal for the time $t_{\text{off}}$.
#
#

# + slideshow={"slide_type": "-"}
npix = s_space.size # whole observation time
l = int(npix * 0.2) # time when the instrument is turned off
h = int(npix * 0.4) # time when the instrument is turned on again
mask = np.zeros(s_space.shape, bool) # initialising a new array for the whole time frame
mask[l:h] = 1 # defining the mask
mask = ift.makeField(s_space, mask) # turning the numpy array into a nifty field
R = ift.MaskOperator(mask) # defining the response operator which masks the places where mask == 1
# -

# ### Synthetic Data
# As in the Wiener filter example with complete data, we are generating some synthetic data now.

N = ift.ScalingOperator(R.target, noise_amplitude**2, float) # defining the noise covariance
n = N.draw_sample() # drawing a noise sample
d = R(s) + n # measuring the signal sample with additional noise

# + slideshow={"slide_type": "skip"}
#D_inv = S.inverse + R.adjoint @ N.inverse @ R
D_inv = ift.SamplingEnabler(ift.SandwichOperator.make(cheese=N.inverse, bun=R), S.inverse, ic, S.inverse)
D_inv = ift.InversionEnabler(D_inv, ic)
D = D_inv.inverse
j = R.adjoint(N.inverse(d))
m = D(j)
# -

n_samples = 200
sc = ift.StatCalculator()
for _ in range(n_samples):
    sample = m + D.draw_sample()
    sc.add(sample)
m_std = sc.var.sqrt()

# + slideshow={"slide_type": "skip"}
plt.axvspan(l, h, facecolor='0.8',alpha=0.5)
plt.fill_between(range(m.size), (m - m_std).val, (m + m_std).val, facecolor='0.5', alpha=0.5, label="Sample std")
plt.plot(sc.mean.val, 'k--', label="Sample mean")
plt.plot(s.val, 'r', label="Signal", alpha=1, linewidth=2)

tmp = R.adjoint(d).val_rw()
tmp[l:h] = np.nan
plt.plot(tmp, 'k.', label="Data")

plt.plot(m.val, 'k', label="Reconstruction", linewidth=2)
plt.title("Reconstruction of incomplete data")
plt.legend()
plt.show()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Wiener Filter standardized
# -
sqrt_pspec = S_k(ift.full(S_k.domain, 1.)).sqrt()
trafo = HT.adjoint @ ift.makeOp(sqrt_pspec)
R2 = R @ trafo
j2 = R2.adjoint(N.inverse(d))
identity = ift.Operator.identity_operator(R2.domain)
Dinv = ift.InversionEnabler(identity + R2.adjoint @ N.inverse @ R2, ic)
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


