import numpy as np
import nifty5 as ift


# TODO: MAKE RESPONSE MPI COMPATIBLE OR USE LOS RESPONSE INSTEAD

class CustomResponse(ift.LinearOperator):
    """
    A custom operator that measures a specific points`

    An operator that is a delta measurement at certain points
    """
    def __init__(self, domain, data_points):
        self._domain = ift.DomainTuple.make(domain)
        self._points = data_points
        data_shape = ift.Field.full(domain, 0.).to_global_data()[data_points]\
            .shape
        self._target = ift.DomainTuple.make(ift.UnstructuredDomain(data_shape))

    def _times(self, x):
        d = np.zeros(self._target.shape, dtype=np.float64)
        d += x.to_global_data()[self._points]
        return ift.from_global_data(self._target, d)

    def _adjoint_times(self, d):
        x = np.zeros(self._domain.shape, dtype=np.float64)
        x[self._points] += d.to_global_data()
        return ift.from_global_data(self._domain, x)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._times(x) if mode == self.TIMES else self._adjoint_times(x)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES


if __name__ == "__main__":
    np.random.seed(43)
    # Set up physical constants
    # Total length of interval or volume the field lives on, e.g. in meters
    L = 2.
    # Typical distance over which the field is correlated (in same unit as L)
    correlation_length = 0.3
    # Variance of field in position space sqrt(<|s_x|^2>) (in same unit as s)
    field_variance = 2.
    # Smoothing length of response (in same unit as L)
    response_sigma = 0.01
    # typical noise amplitude of the measurement
    noise_level = 0.

    # Define resolution (pixels per dimension)
    N_pixels = 256

    # Set up derived constants
    k_0 = 1./correlation_length
    # defining a power spectrum with the right correlation length
    # we later set the field variance to the desired value
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

    Rx = CustomResponse(s_space, [np.arange(0, N_pixels, 5)[:, np.newaxis],
                                  np.arange(0, N_pixels, 2)[np.newaxis, :]])
    ift.extra.consistency_check(Rx)
    a = ift.Field.from_random('normal', s_space)
    b = ift.Field.from_random('normal', Rx.target)
    R = Rx * HT

    noiseless_data = R(sh)
    N = ift.ScalingOperator(noise_level**2, R.target)
    n = N.draw_sample()

    d = noiseless_data + n

    # Wiener filter

    IC = ift.GradientNormController(name="inverter", iteration_limit=1000,
                                    tol_abs_gradnorm=0.0001)
    inverter = ift.ConjugateGradient(controller=IC)
    # setting up measurement precision matrix M
    M = (ift.SandwichOperator.make(R.adjoint, Sh) + N)
    M = ift.InversionEnabler(M, inverter)
    m = Sh(R.adjoint(M.inverse_times(d)))

    # Plotting
    backprojection = Rx.adjoint(d)
    reweighted_backprojection = (backprojection / backprojection.max() *
                                 HT(sh).max())
    zmax = max(HT(sh).max(), reweighted_backprojection.max(), HT(m).max())
    zmin = min(HT(sh).min(), reweighted_backprojection.min(), HT(m).min())
    plotdict = {"colormap": "Planck-like", "zmax": zmax, "zmin": zmin}
    ift.plot(HT(sh), name="mock_signal.png", **plotdict)
    ift.plot(backprojection, name="backprojected_data.png", **plotdict)
    ift.plot(HT(m), name="reconstruction.png", **plotdict)
