import numpy as np
import nifty4 as ift


if __name__ == "__main__":
    np.random.seed(43)
    # Set up physical constants
    # Total length of interval or volume the field lives on, e.g. in meters
    L = 2.
    # Typical distance over which the field is correlated (in same unit as L)
    correlation_length = 0.1
    # Variance of field in position space sqrt(<|s_x|^2>) (in same unit as s)
    field_variance = 2.
    # Smoothing length of response (in same unit as L)
    response_sigma = 0.01
    # typical noise amplitude of the measurement
    noise_level = 1.

    # Define resolution (pixels per dimension)
    N_pixels = 256

    # Set up derived constants
    k_0 = 1./correlation_length
    #defining a power spectrum with the right correlation length
    #we later set the field variance to the desired value
    unscaled_pow_spec = (lambda k: 1. / (1 + k/k_0) ** 4)
    pixel_width = L/N_pixels

    # Set up the geometry
    s_space = ift.RGSpace([N_pixels, N_pixels], distances=pixel_width)
    h_space = s_space.get_default_codomain()
    s_var = ift.get_signal_variance(unscaled_pow_spec, h_space)
    pow_spec = (lambda k: unscaled_pow_spec(k)/s_var*field_variance**2)
    
    HT = ift.HarmonicTransformOperator(h_space, s_space)

    # Create mock data

    Sh = ift.create_power_operator(h_space, power_spectrum=pow_spec)
    sh = Sh.draw_sample()

    R = HT*ift.create_harmonic_smoothing_operator((h_space,), 0,
                                                  response_sigma)
    noiseless_data = R(sh)
    N = ift.ScalingOperator(noise_level**2, s_space)
    n = N.draw_sample()

    d = noiseless_data + n

    # Wiener filter

    j = R.adjoint_times(N.inverse_times(d))
    IC = ift.GradientNormController(name="inverter", iteration_limit=500,
                                    tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=IC)
    D = (ift.SandwichOperator(R, N.inverse) + Sh.inverse).inverse
    D = ift.InversionEnabler(D, inverter, approximation=Sh)
    m = D(j)

    # Plotting
    d_field = d.cast_domain(s_space)
    zmax = max(HT(sh).max(), d_field.max(), HT(m).max())
    zmin = min(HT(sh).min(), d_field.min(), HT(m).min())
    plotdict = {"colormap": "Planck-like", "zmax": zmax, "zmin": zmin}
    ift.plot(HT(sh), name="mock_signal.png", **plotdict)
    ift.plot(d_field, name="data.png", **plotdict)
    ift.plot(HT(m), name="reconstruction.png", **plotdict)
