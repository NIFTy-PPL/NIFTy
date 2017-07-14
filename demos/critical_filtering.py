from nifty import *
from nifty.library.wiener_filter import WienerFilterEnergy
from nifty.library.critical_filter import CriticalPowerEnergy
import plotly.offline as pl
import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(42)


def plot_parameters(m,t,p, p_d):

    x = log(t.domain[0].kindex)
    m = fft.adjoint_times(m)
    m = m.val.get_full_data().real
    t = t.val.get_full_data().real
    p = p.val.get_full_data().real
    p_d = p_d.val.get_full_data().real
    pl.plot([go.Heatmap(z=m)], filename='map.html')
    pl.plot([go.Scatter(x=x,y=t), go.Scatter(x=x ,y=p), go.Scatter(x=x, y=p_d)], filename="t.html")


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
    p_space = PowerSpace(h_space, logarithmic=True,
                         distribution_strategy=distribution_strategy)

    # Choosing the prior correlation structure and defining correlation operator
    p_spec = (lambda k: (.5 / (k + 1) ** 3))
    S = create_power_operator(h_space, power_spectrum=p_spec,
                              distribution_strategy=distribution_strategy)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = Field(p_space,  val=p_spec,
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)


    # Choosing the measurement instrument
    Instrument = SmoothingOperator(s_space, sigma=0.01)
    # Instrument = DiagonalOperator(s_space, diagonal=1.)
    # Instrument._diagonal.val[200:400, 200:400] = 0
    #Instrument._diagonal.val[64:512-64, 64:512-64] = 0


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

    # The information source
    j = R.adjoint_times(N.inverse_times(d))
    realized_power = log(sh.power_analyze(binbounds=p_space.binbounds))
    data_power = log(fft(d).power_analyze(binbounds=p_space.binbounds))
    d_data = d.val.get_full_data().real
    #if rank == 0:
    #    pl.plot([go.Heatmap(z=d_data)], filename='data.html')

    #  minimization strategy

    def convergence_measure(a_energy, iteration): # returns current energy
        x = a_energy.value
        print (x, iteration)


    minimizer1 = RelaxedNewton(convergence_tolerance=1e-2,
                              convergence_level=2,
                              iteration_limit=5,
                              callback=convergence_measure)

    minimizer2 = VL_BFGS(convergence_tolerance=1e-3,
                       iteration_limit=20,
                       callback=convergence_measure,
                       max_history_length=10)
    minimizer3 = SteepestDescent(convergence_tolerance=1e-3,
                       iteration_limit=70,
                       callback=convergence_measure)

    # Setting starting position
    flat_power = Field(p_space,val=1e-8)
    m0 = flat_power.power_synthesize(real_signal=True)

    t0 = Field(p_space, val=log(1./(1+p_space.kindex)**2))



    for i in range(500):
        S0 = create_power_operator(h_space, power_spectrum=exp(t0),
                              distribution_strategy=distribution_strategy)

        # Initializing the  nonlinear Wiener Filter energy
        map_energy = WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S0)
        # Solving the Wiener Filter analytically
        D0 = map_energy.curvature
        m0 = D0.inverse_times(j)
        # Initializing the power energy with updated parameters
        power_energy = CriticalPowerEnergy(position=t0, m=m0, D=D0, smoothness_prior=10., samples=3)

        (power_energy, convergence) = minimizer3(power_energy)


        # Setting new power spectrum
        t0.val  = power_energy.position.val.real

        # Plotting current estimate
        print i
        #if i%50 == 0:
        #    plot_parameters(m0,t0,log(sp), data_power)


