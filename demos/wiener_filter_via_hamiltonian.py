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
    sp = ift.Field(p_space, val=p_spec(p_space.k_lengths))
    sh = sp.power_synthesize(real_signal=True)
    ss = fft.adjoint_times(sh)

    # Choosing the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.05)
    Instrument = ift.DiagonalOperator(ift.Field(s_space, 1.))
#    Instrument._diagonal.val[200:400, 200:400] = 0

    # Adding a harmonic transformation to the instrument
    R = AdjointFFTResponse(fft, Instrument)
    signal_to_noise = 1.
    ndiag = ift.Field(s_space, ss.var()/signal_to_noise).weight(1)
    N = ift.DiagonalOperator(ndiag)
    n = ift.Field.from_random(domain=s_space,
                          random_type='normal',
                          std=ss.std()/np.sqrt(signal_to_noise),
                          mean=0)

    # Creating the mock data
    d = R(sh) + n
    j = R.adjoint_times(N.inverse_times(d))

    # Choosing the minimization strategy

    ctrl = ift.DefaultIterationController(verbose=True,tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=ctrl)
    # Setting starting position
    m0 = ift.Field(h_space, val=.0)

    # Initializing the Wiener Filter energy
    energy = ift.library.WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S,
                                inverter=inverter)
    D0 = energy.curvature

    # Solving the problem analytically
    m0 = D0.inverse_times(j)

    sample_variance = ift.Field(sh.domain, val=0.)
    sample_mean = ift.Field(sh.domain, val=0.)

    # sampling the uncertainty map
    n_samples = 50
    for i in range(n_samples):
        sample = fft(ift.sugar.generate_posterior_sample(0., D0))
        sample_variance += sample**2
        sample_mean += sample
    variance = (sample_variance - sample_mean**2)/n_samples
