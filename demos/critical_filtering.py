import numpy as np
import nifty2go as ift

np.random.seed(42)
#np.seterr(all="raise",under="ignore")

def plot_parameters(m, t, p, p_d):
    x = np.log(t.domain[0].k_lengths)
    m = fft.adjoint_times(m)
    t = t.val.real
    p = p.val.real
    p_d = p_d.val.real
    ift.plotting.plot(m.real, name='map.pdf')
    #pl.plot([go.Scatter(x=x, y=t), go.Scatter(x=x, y=p),
    #         go.Scatter(x=x, y=p_d)], filename="t.html", auto_open=False)


class AdjointFFTResponse(ift.LinearOperator):
    def __init__(self, FFT, R, default_spaces=None):
        super(AdjointFFTResponse, self).__init__(default_spaces)
        self._domain = FFT.target
        self._target = R.target
        self.R = R
        self.FFT = FFT

    def _times(self, x, spaces=None):
        return self.R(self.FFT.adjoint_times(x))

    def _adjoint_times(self, x, spaces=None):
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
    # s_space = ift.HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    fft = ift.FFTOperator(s_space)
    h_space = fft.target[0]

    # Set up power space
    p_space = ift.PowerSpace(h_space,binbounds=ift.PowerSpace.useful_binbounds(h_space,logarithmic=True))

    # Choose the prior correlation structure and defining correlation operator
    p_spec = (lambda k: (.5 / (k + 1) ** 3))
    S = ift.create_power_operator(h_space, power_spectrum=p_spec)

    # Draw a sample sh from the prior distribution in harmonic space
    sp = ift.Field(p_space,  val=p_spec(p_space.k_lengths))
    sh = sp.power_synthesize(real_signal=True)

    # Choose the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.01)
    Instrument = ift.DiagonalOperator(s_space, diagonal=1.)
    # Instrument._diagonal.val[200:400, 200:400] = 0
    # Instrument._diagonal.val[64:512-64, 64:512-64] = 0

    # Add a harmonic transformation to the instrument
    R = AdjointFFTResponse(fft, Instrument)

    noise = 1.
    N = ift.DiagonalOperator(s_space, ift.Field(s_space,noise).weight(1))
    n = ift.Field.from_random(domain=s_space,
                          random_type='normal',
                          std=np.sqrt(noise),
                          mean=0)

    # Create mock data
    d = R(sh) + n

    # The information source
    j = R.adjoint_times(N.inverse_times(d))
    realized_power = ift.log(sh.power_analyze(binbounds=p_space.binbounds))
    data_power = ift.log(fft(d).power_analyze(binbounds=p_space.binbounds))
    d_data = d.val.real
    ift.plotting.plot(d.real, name="data.pdf")

    IC1 = ift.DefaultIterationController(verbose=True,iteration_limit=100,tol_abs_gradnorm=0.1)
    minimizer1 = ift.RelaxedNewton(IC1)
    IC2 = ift.DefaultIterationController(verbose=True,iteration_limit=100,tol_abs_gradnorm=0.1)
    minimizer2 = ift.VL_BFGS(IC2, max_history_length=20)
    IC3 = ift.DefaultIterationController(verbose=True,iteration_limit=100,tol_abs_gradnorm=0.1)
    minimizer3 = ift.SteepestDescent(IC3)

    # Set starting position
    flat_power = ift.Field(p_space, val=1e-8)
    m0 = flat_power.power_synthesize(real_signal=True)

    def ps0(k):
        return (1./(1.+k)**2)
    t0 = ift.Field(p_space, val=np.log(1./(1+p_space.k)_lengths)**2))

    for i in range(500):
        S0 = ift.create_power_operator(h_space, power_spectrum=ps0)

        # Initialize non-linear Wiener Filter energy
        ICI = ift.DefaultIterationController(verbose=False,iteration_limit=500,tol_abs_gradnorm=0.1)
        map_inverter = ift.ConjugateGradient(controller=ICI)
        map_energy = ift.library.WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S0, inverter=map_inverter)
        # Solve the Wiener Filter analytically
        D0 = map_energy.curvature
        m0 = D0.inverse_times(j)
        # Initialize power energy with updated parameters
        ICI2 = ift.DefaultIterationController(name="powI",verbose=True,iteration_limit=200,tol_abs_gradnorm=1e-5)
        power_inverter = ift.ConjugateGradient(controller=ICI2)
        power_energy = ift.library.CriticalPowerEnergy(position=t0, m=m0, D=D0,
                                           smoothness_prior=10., samples=3, inverter=power_inverter)

        (power_energy, convergence) = minimizer1(power_energy)

        # Set new power spectrum
        t0 = power_energy.position.real

        # Plot current estimate
        print(i)
        if i % 5 == 0:
            plot_parameters(m0, t0, ift.log(sp), data_power)
