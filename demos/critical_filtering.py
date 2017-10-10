import numpy as np
import nifty as ift
from nifty.library.critical_filter import CriticalPowerEnergy
from nifty.library.wiener_filter import WienerFilterEnergy

import plotly.graph_objs as go
import plotly.offline as pl
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(44)


def plot_parameters(m, t, p, p_sig,p_d):

    x = np.log(t.domain[0].kindex)
    m = fft.adjoint_times(m)
    m = m.val.get_full_data().real
    t = t.val.get_full_data().real
    p = p.val.get_full_data().real
    pd_sig = p_sig.val.get_full_data()
    p_d = p_d.val.get_full_data().real
    pl.plot([go.Heatmap(z=m)], filename='map.html', auto_open=False)
    pl.plot([go.Scatter(x=x, y=t), go.Scatter(x=x, y=p),
             go.Scatter(x=x, y=p_d),go.Scatter(x=x, y=pd_sig)], filename="t.html", auto_open=False)


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

    distribution_strategy = 'not'

    # Set up position space
    dist = 1/128. *0.1
    s_space = ift.RGSpace([128, 128], distances=[dist,dist])
    # s_space = ift.HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    fft = ift.FFTOperator(s_space)
    h_space = fft.target[0]

    # Set up power space
    p_space = ift.PowerSpace(h_space,
                             binbounds=ift.PowerSpace.useful_binbounds(
                                       h_space, logarithmic=True),
                             distribution_strategy=distribution_strategy)

    # Choose the prior correlation structure and defining correlation operator
    p_spec = (lambda k: (.5 / (k + 1) ** 3))
    # p_spec = (lambda k: 1)
    S = ift.create_power_operator(h_space, power_spectrum=p_spec,
                                  distribution_strategy=distribution_strategy)

    # Draw a sample sh from the prior distribution in harmonic space
    sp = ift.Field(p_space,  val=p_spec,
                   distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)

    # Choose the measurement instrument
    # Instrument = ift.SmoothingOperator(s_space, sigma=0.01)
    Instrument = ift.DiagonalOperator(s_space, diagonal=1.)
    # Instrument._diagonal.val[200:400, 200:400] = 0
    # Instrument._diagonal.val[64:512-64, 64:512-64] = 0

    # Add a harmonic transformation to the instrument
    R = AdjointFFTResponse(fft, Instrument)

    noise = 100.
    ndiag = ift.Field(s_space, noise).weight(1)
    N = ift.DiagonalOperator(s_space, ndiag)
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
    d_data = d.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=d_data)], filename='data.html', auto_open=False)

    #  Minimization strategy
    def convergence_measure(a_energy, iteration):  # returns current energy
        x = a_energy.value
        print(x, iteration)

    IC1 = ift.GradientNormController(iteration_limit=5,
                                     tol_abs_gradnorm=0.1)
    minimizer1 = ift.RelaxedNewton(IC1)
    IC2 = ift.GradientNormController(iteration_limit=30,
                                     tol_abs_gradnorm=0.1)
    minimizer2 = ift.VL_BFGS(IC2, max_history_length=20)
    IC3 = ift.GradientNormController(iteration_limit=100,
                                     tol_abs_gradnorm=0.1)
    minimizer3 = ift.SteepestDescent(IC3)
    # Set starting position
    flat_power = ift.Field(p_space, val=1e-8)
    m0 = flat_power.power_synthesize(real_signal=True)

    # t0 = ift.Field(p_space, val=np.log(1./(1+p_space.kindex)**2))
    t0 = ift.Field(p_space, val=-5)

    for i in range(500):
        S0 = ift.create_power_operator(h_space, power_spectrum=ift.exp(t0),
                                       distribution_strategy=distribution_strategy)

        # Initialize non-linear Wiener Filter energy
        map_energy = WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S0)
        # Solve the Wiener Filter analytically
        D0 = map_energy.curvature
        m0 = D0.inverse_times(j)
        # Initialize power energy with updated parameters
        power_energy = CriticalPowerEnergy(position=t0, m=m0, D=D0,
                                           smoothness_prior=1e-15, samples=5)

        (power_energy, convergence) = minimizer1(power_energy)

        # Set new power spectrum
        t0.val = power_energy.position.val.real

        # Plot current estimate
        print(i)
        if i % 1 == 0:
            plot_parameters(sh, t0, ift.log(sp), ift.log(sh.power_analyze(binbounds=p_space.binbounds)),data_power)
            print ift.log(sh.power_analyze(binbounds=p_space.binbounds)).val - t0.val
