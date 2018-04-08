# Program to generate figures of article "Information theory for fields"
# by Torsten Ensslin, Annalen der Physik, submitted to special edition
# "Physics of Information" in April 2018

import numpy as np
import nifty4 as ift
import matplotlib.pyplot as plt


class Exp3(object):
    def __call__(self, x):
        return ift.exp(3*x)

    def derivative(self, x):
        return 3*ift.exp(3*x)


if __name__ == "__main__":
    np.random.seed(20)

    # Set up physical constants
    nu = 1.         # excitation field level
    kappa = 10.     # diffusion constant
    eps = 1e-8      # small number to tame zero mode
    sigma_n = 0.2   # noise level
    sigma_n2 = sigma_n**2
    L = 1.          # Total length of interval or volume the field lives on
    nprobes = 1000  # Number of probes for uncertainty quantification

    # Define resolution (pixels per dimension)
    N_pixels = 1024

    # Define data gaps
    N1a = int(0.6*N_pixels)
    N1b = int(0.64*N_pixels)
    N2a = int(0.67*N_pixels)
    N2b = int(0.8*N_pixels)

    # Set up derived constants
    amp = nu/(2*kappa)  # spectral normalization
    pow_spec = lambda k: amp / (eps + k**2)
    lambda2 = 2*kappa*sigma_n2/nu  # resulting correlation length squared
    lambda1 = np.sqrt(lambda2)
    pixel_width = L/N_pixels
    x = np.arange(0, 1, pixel_width)

    # Set up the geometry
    s_domain = ift.RGSpace([N_pixels], distances=pixel_width)
    h_domain = s_domain.get_default_codomain()
    HT = ift.HarmonicTransformOperator(h_domain, s_domain)
    aHT = HT.adjoint

    # Create mock signal
    Phi_h = ift.create_power_operator(h_domain, power_spectrum=pow_spec)
    phi_h = Phi_h.draw_sample()
    # remove zero mode
    glob = phi_h.to_global_data()
    glob[0] = 0.
    phi_h = ift.Field.from_global_data(phi_h.domain, glob)
    phi = HT(phi_h)

    # Setting up an exemplary response
    GeoRem = ift.GeometryRemover(s_domain)
    d_domain = GeoRem.target[0]
    mask = np.ones(d_domain.shape)
    mask[N1a:N1b] = 0.
    mask[N2a:N2b] = 0.
    fmask = ift.Field.from_global_data(d_domain, mask)
    Mask = ift.DiagonalOperator(fmask)
    R0 = Mask*GeoRem
    R = R0*HT

    # Linear measurement scenario
    N = ift.ScalingOperator(sigma_n2, d_domain)  # Noise covariance
    n = Mask(N.draw_sample())  # seting the noise to zero in masked region
    d = R(phi_h) + n

    # Wiener filter
    j = R.adjoint_times(N.inverse_times(d))
    IC = ift.GradientNormController(name="inverter", iteration_limit=500,
                                    tol_abs_gradnorm=1e-3)
    inverter = ift.ConjugateGradient(controller=IC)
    D = (ift.SandwichOperator(R, N.inverse) + Phi_h.inverse).inverse
    D = ift.InversionEnabler(D, inverter, approximation=Phi_h)
    m = HT(D(j))

    # Uncertainty
    D = ift.SandwichOperator(aHT, D)  # real space propagator
    Dhat = ift.probe_with_posterior_samples(D.inverse, None,
                                            nprobes=nprobes)[1]
    sig = ift.sqrt(Dhat)

    # Plotting
    x_mod = np.where(mask > 0, x, None)
    plt.rcParams["text.usetex"] = True
    plt.fill_between(x, m.val-sig.val, m.val+sig.val, color='pink',
                     alpha=None)
    plt.plot(x, phi.to_global_data(), label=r"$\varphi$", color='black')
    plt.scatter(x_mod, d.to_global_data(), label=r'$d$', s=1, color='blue',
                alpha=0.5)
    plt.plot(x, m.to_global_data(), label=r'$m$', color='red')
    plt.xlim([0, L])
    plt.ylim([-1, 1])
    plt.title('Wiener filter reconstruction')
    plt.legend()
    plt.savefig('Wiener_filter.pdf')
    plt.close()

    nonlin = Exp3()
    lam = R0(nonlin(HT(phi_h)))
    data = ift.Field.from_local_data(
        d_domain, np.random.poisson(lam.local_data).astype(np.float64))

    # initial guess
    psi0 = ift.Field.full(h_domain, 1e-7)
    energy = ift.library.PoissonEnergy(psi0, data, R0, nonlin, HT, Phi_h,
                                       inverter)
    IC1 = ift.GradientNormController(name="IC1", iteration_limit=200,
                                     tol_abs_gradnorm=1e-4)
    minimizer = ift.RelaxedNewton(IC1)
    energy = minimizer(energy)[0]

    var = ift.probe_with_posterior_samples(energy.curvature, HT, nprobes)[1]
    sig = ift.sqrt(var)

    m = HT(energy.position)
    phi = HT(phi_h)
    plt.rcParams["text.usetex"] = True
    c1 = nonlin(m-sig).to_global_data()
    c2 = nonlin(m+sig).to_global_data()
    plt.fill_between(x, c1, c2, color='pink', alpha=None)
    plt.plot(x, nonlin(phi).to_global_data(), label=r"$e^{3\varphi}$",
             color='black')
    plt.scatter(x_mod, data.to_global_data(), label=r'$d$', s=1, color='blue',
                alpha=0.5)
    plt.plot(x, nonlin(m).to_global_data(),
             label=r'$e^{3\varphi_\mathrm{cl}}$', color='red')
    plt.xlim([0, L])
    plt.ylim([-0.1, 7.5])
    plt.title('Poisson log-normal reconstruction')
    plt.legend()
    plt.savefig('Poisson.pdf')
