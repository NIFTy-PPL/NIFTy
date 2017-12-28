import numpy as np
import nifty2go as ift


if __name__ == "__main__":
    np.random.seed(43)
    # Set up physical constants
    # Total length of interval or volume the field lives on, e.g. in meters
    L = 2.
    # Typical distance over which the field is correlated (in same unit as L)
    correlation_length = 0.1
    # Variance of field in position space sqrt(<|s_x|^2>) (in unit of s)
    field_variance = 2.
    # Smoothing length of response (in same unit as L)
    response_sigma = 0.01

    # Define resolution (pixels per dimension)
    N_pixels = 256

    # Set up derived constants
    k_0 = 1./correlation_length
    # Note that field_variance**2 = a*k_0/4. for this analytic form of power
    # spectrum
    a = field_variance**2/k_0*4.
    pow_spec = (lambda k: a / (1 + k/k_0) ** 4)
    pixel_width = L/N_pixels

    # Set up the geometry
    s_space = ift.RGSpace([N_pixels, N_pixels], distances=pixel_width)
    fft = ift.FFTOperator(s_space)
    h_space = fft.target[0]
    p_space = ift.PowerSpace(h_space)

    # Create mock data

    Sh = ift.create_power_operator(h_space, power_spectrum=pow_spec)

    sp = ift.PS_field(p_space, pow_spec)
    sh = ift.power_synthesize(sp, real_signal=True)
    ss = fft.inverse_times(sh)

    R = ift.FFTSmoothingOperator(s_space, sigma=response_sigma)

    signal_to_noise = 1
    diag = ift.Field(s_space, ss.var()/signal_to_noise).weight(1)
    N = ift.DiagonalOperator(diag)
    n = ift.Field.from_random(domain=s_space,
                              random_type='normal',
                              std=ss.std()/np.sqrt(signal_to_noise),
                              mean=0)

    d = R(ss) + n

    # Wiener filter

    j = R.adjoint_times(N.inverse_times(d))
    IC = ift.GradientNormController(verbose=True, iteration_limit=500,
                                    tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=IC)
    S_inv = fft.adjoint*Sh.inverse*fft
    D = (R.adjoint*N.inverse*R + S_inv).inverse
    # MR FIXME: we can/should provide a preconditioner here as well!
    D = ift.InversionEnabler(D, inverter)
    m = D(j)
