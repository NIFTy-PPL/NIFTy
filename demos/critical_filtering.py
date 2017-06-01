
from nifty import *

import plotly.offline as pl
import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(232)


def plot_parameters(m,t,t_true, t_real, t_d):
    m = fft.adjoint_times(m)
    m_data = m.val.get_full_data().real
    t_data = t.val.get_full_data().real
    t_d_data = t_d.val.get_full_data().real
    t_true_data = t_true.val.get_full_data().real
    t_real_data = t_real.val.get_full_data().real
    pl.plot([go.Heatmap(z=m_data)], filename='map.html')
    pl.plot([go.Scatter(y=t_data), go.Scatter(y=t_true_data),
             go.Scatter(y=t_real_data), go.Scatter(y=t_d_data)], filename="t.html")

class AdjointFFTResponse(LinearOperator):
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
    s_space = RGSpace([128,128])
    # s_space = HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    fft = FFTOperator(s_space)
    h_space = fft.target[0]

    # Setting up power space
    p_space = PowerSpace(h_space, logarithmic=False,
                         distribution_strategy=distribution_strategy, nbin=128)

    # Choosing the prior correlation structure and defining correlation operator
    pow_spec = (lambda k: (.05 / (k + 1) ** 2))
    S = create_power_operator(h_space, power_spectrum=pow_spec,
                              distribution_strategy=distribution_strategy)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = Field(p_space,  val=lambda z: pow_spec(z)**(1./2),
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)


    # Choosing the measurement instrument
#    Instrument = SmoothingOperator(s_space, sigma=0.01)
    Instrument = DiagonalOperator(s_space, diagonal=1.)
#    Instrument._diagonal.val[200:400, 200:400] = 0


    #Adding a harmonic transformation to the instrument
    R = AdjointFFTResponse(fft, Instrument)

    noise = 1.
    N = DiagonalOperator(s_space, diagonal=noise, bare=True)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=sqrt(noise),
                          mean=0)

    # Creating the mock data
    d = R(sh) + n

    realized_power = log(sh.power_analyze(logarithmic=p_space.config["logarithmic"],
                                          nbin=p_space.config["nbin"])**2)
    data_power  = log(fft(d).power_analyze(logarithmic=p_space.config["logarithmic"],
                                          nbin=p_space.config["nbin"])**2)
    d_data = d.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=d_data)], filename='data.html')

    #  minimization strategy

    def convergence_measure(a_energy, iteration): # returns current energy
        x = a_energy.value
        print (x, iteration)


    minimizer1 = RelaxedNewton(convergence_tolerance=0,
                              convergence_level=1,
                              iteration_limit=2,
                              callback=convergence_measure)
    minimizer2 = VL_BFGS(convergence_tolerance=0,
                       iteration_limit=50,
                       callback=convergence_measure,
                       max_history_length=3)

    # Setting starting position
    flat_power = Field(p_space,val=10e-8)
    m0 = flat_power.power_synthesize(real_signal=True)

    t0 = Field(p_space, val=log(1./(1+p_space.kindex)**2))
    #t0 = data_power
    S0 = create_power_operator(h_space, power_spectrum=exp(t0),
                               distribution_strategy=distribution_strategy)


    for i in range(100):
        S0 = create_power_operator(h_space, power_spectrum=exp(t0),
                              distribution_strategy=distribution_strategy)

        # Initializing the  nonlinear Wiener Filter energy
        map_energy = WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S0)
        # Minimization with chosen minimizer
        map_energy = map_energy.analytic_solution()

        # Updating parameters for correlation structure reconstruction
        m0 = map_energy.position
        D0 = map_energy.curvature
        # Initializing the power energy with updated parameters
        power_energy = CriticalPowerEnergy(position=t0, m=m0, D=D0, sigma=10., samples=3)
        if i > 0:
            (power_energy, convergence) = minimizer1(power_energy)
        else:
            (power_energy, convergence) = minimizer2(power_energy)
        # Setting new power spectrum
        t0 = power_energy.position
        t0.val[-1] = t0.val[-2]
        # Plotting current estimate
        plot_parameters(m0,t0,log(sp**2),realized_power, data_power)

    # Transforming fields to position space for plotting

    ss = fft.adjoint_times(sh)
    m = fft.adjoint_times(map_energy.position)


    # Plotting

    d_data = d.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=d_data)], filename='data.html')

    tt_data = power_energy.position.val.get_full_data().real
    t_data = log(sp**2).val.get_full_data().real
    if rank == 0:
        pl.plot([go.Scatter(y=t_data),go.Scatter(y=tt_data)], filename="t.html")
    ss_data = ss.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=ss_data)], filename='ss.html')

    sh_data = sh.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=sh_data)], filename='sh.html')


    m_data = m.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=m_data)], filename='map.html')
