import numpy as np
import nifty2go as ift


# Note that the constructor of PropagatorOperator takes as arguments the
# response R and noise covariance N operating on signal space and signal
# covariance operating on harmonic space.
class PropagatorOperator(ift.InversionEnabler, ift.EndomorphicOperator):
    def __init__(self, R, N, Sh, inverter):
        ift.InversionEnabler.__init__(self, inverter)
        ift.EndomorphicOperator.__init__(self)

        self.R = R
        self.N = N
        self.Sh = Sh
        self.fft = ift.FFTOperator(R.domain, target=Sh.domain)

    def _inverse_times(self, x):
        return self.R.adjoint_times(self.N.inverse_times(self.R(x))) \
               + self.fft.adjoint_times(self.Sh.inverse_times(self.fft(x)))

    @property
    def domain(self):
        return self.R.domain

    @property
    def unitary(self):
        return False

    @property
    def self_adjoint(self):
        return True


if __name__ == "__main__":
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

    sp = ift.Field(p_space, val=pow_spec(p_space.k_lengths))
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
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=IC)
    D = PropagatorOperator(Sh=Sh, N=N, R=R, inverter=inverter)

    m = D(j)
