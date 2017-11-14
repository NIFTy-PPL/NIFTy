import nifty2go as ift
import numpy as np

np.random.seed(42)


class AdjointFFTResponse(ift.LinearOperator):
    def __init__(self, FFT, R):
        super(AdjointFFTResponse, self).__init__()
        self._domain = FFT.target
        self._target = R.target
        self.R = R
        self.FFT = FFT

    def _times(self, x):
        return self.R(self.FFT.adjoint_times(x))

    def _adjoint_times(self, x):
        return self.FFT(self.R.adjoint_times(x))

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False


if __name__ == "__main__":
    # Set up position space
    s_space = ift.RGSpace([128, 128])
    # s_space = HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    fft = ift.FFTOperator(s_space)
    h_space = fft.target[0]

    # Setting up power space
    p_space = ift.PowerSpace(h_space)

    # Choosing the prior correlation structure and defining
    # correlation operator
    p_spec = (lambda k: (42 / (k + 1) ** 3))

    S = ift.create_power_operator(h_space, power_spectrum=p_spec)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = ift.PS_field(p_space, p_spec)
    sh = ift.power_synthesize(sp, real_signal=True)
    ss = fft.adjoint_times(sh)

    # Choosing the measurement instrument
    # Instrument = ift.FFTSmoothingOperator(s_space, sigma=0.05)
    diag = np.ones(s_space.shape)
    diag[20:80, 20:80] = 0
    diag = ift.Field(s_space, ift.dobj.from_global_data(diag))
    Instrument = ift.DiagonalOperator(diag)

    # Adding a harmonic transformation to the instrument
    R = AdjointFFTResponse(fft, Instrument)
    signal_to_noise = 1.
    ndiag = ift.Field.full(s_space, ss.var()/signal_to_noise)
    N = ift.DiagonalOperator(ndiag.weight(1))
    n = ift.Field.from_random(domain=s_space,
                              random_type='normal',
                              std=ss.std()/np.sqrt(signal_to_noise),
                              mean=0)

    # Creating the mock data
    d = R(sh) + n
    j = R.adjoint_times(N.inverse_times(d))

    # Choosing the minimization strategy

    ctrl = ift.GradientNormController(verbose=True, tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=ctrl)
    controller = ift.GradientNormController(verbose=True, tol_abs_gradnorm=0.1)
    minimizer = ift.RelaxedNewton(controller=controller)
    m0 = ift.Field.zeros(h_space)
    # Initializing the Wiener Filter energy
    energy = ift.library.WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S,
                                            inverter=inverter)

    energy, convergence = minimizer(energy)
    m = energy.position
    D = energy.curvature
    ift.plotting.plot(ss, name="signal.pdf", colormap="Planck-like")
    ift.plotting.plot(fft.inverse_times(m), name="m.pdf",
                      colormap="Planck-like")

    # sampling the uncertainty map
    sample_variance = ift.Field.zeros(s_space)
    sample_mean = ift.Field.zeros(s_space)

    n_samples = 50
    for i in range(n_samples):
        sample = fft.adjoint_times(ift.sugar.generate_posterior_sample(m, D))
        sample_variance += sample**2
        sample_mean += sample
    sample_mean /= n_samples
    sample_variance /= n_samples
    variance = sample_variance - sample_mean**2
    ift.plotting.plot(variance, name="variance.pdf", colormap="Planck-like")
