
from nifty import *

import plotly.offline as pl
import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(62)

class NonlinearResponse(LinearOperator):
    def __init__(self, FFT, Instrument, function, derivative, default_spaces=None):
        super(NonlinearResponse, self).__init__(default_spaces)
        self._domain = FFT.target
        self._target = Instrument.target
        self.function = function
        self.derivative = derivative
        self.I = Instrument
        self.FFT = FFT


    def _times(self, x, spaces=None):
        return self.I(self.function(self.FFT.adjoint_times(x)))

    def _adjoint_times(self, x, spaces=None):
        return self.FFT(self.function(self.I.adjoint_times(x)))

    def derived_times(self, x, position):
        position_derivative = self.derivative(self.FFT.adjoint_times(position))
        return self.I(position_derivative * self.FFT.adjoint_times(x))

    def derived_adjoint_times(self, x, position):
        position_derivative = self.derivative(self.FFT.adjoint_times(position))
        return self.FFT(position_derivative * self.I.adjoint_times(x))

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False

def plot_parameters(m,t,t_true, t_real):
    m = fft.adjoint_times(m)
    m_data = m.val.get_full_data().real
    t_data = t.val.get_full_data().real
    t_true_data = t_true.val.get_full_data().real
    t_real_data = t_real.val.get_full_data().real
    pl.plot([go.Heatmap(z=m_data)], filename='map.html')
    pl.plot([go.Scatter(y=t_data), go.Scatter(y=t_true_data),
             go.Scatter(y=t_real_data)], filename="t.html")

if __name__ == "__main__":

    distribution_strategy = 'not'

    # Set up position space
    s_space = RGSpace([128,128])
    # s_space = HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    fft = FFTOperator(s_space)
    h_space = fft.target[0]

    # Setting up power space
    p_space = PowerSpace(h_space, logarithmic = True,
                         distribution_strategy=distribution_strategy)

    # Choosing the prior correlation structure and defining correlation operator
    pow_spec = (lambda k: (.05 / (k + 1) ** 3))
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

    #   Choosing nonlinearity

    # log-normal model:

    function = exp
    derivative = exp

    # tan-normal model

    # def function(x):
    #     return 0.5 * tanh(x) + 0.5
    # def derivative(x):
    #     return 0.5*(1 - tanh(x)**2)

    # no nonlinearity, Wiener Filter

    # def function(x):
    #     return x
    # def derivative(x):
    #     return 1

    # small quadratic pertubarion

    # def function(x):
    #     return 0.5*x**2 + x
    # def derivative(x):
    #     return x + 1

    # def function(x):
    #     return 0.9*x**4 +0.2*x**2 + x
    # def derivative(x):
    #     return 0.9*4*x**3 + 0.4*x +1
    #

    #Adding a harmonic transformation to the instrument
    R = NonlinearResponse(fft, Instrument, function, derivative)
    noise = .01
    N = DiagonalOperator(s_space, diagonal=noise, bare=True)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=sqrt(noise),
                          mean=0)

    # Creating the mock data
    d = R(sh) + n
    realized_power = log(sh.power_analyze(logarithmic=p_space.config["logarithmic"])**2)
    d_data = d.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=d_data)], filename='data.html')

    # Choosing the minimization strategy

    def convergence_measure(a_energy, iteration): # returns current energy
        x = a_energy.value
        print (x, iteration)

    # minimizer1 = SteepestDescent(convergence_tolerance=0,
    #                            iteration_limit=50,
    #                            callback=convergence_measure)

    minimizer1 = RelaxedNewton(convergence_tolerance=0,
                              convergence_level=1,
                              iteration_limit=5,
                              callback=convergence_measure)
    # minimizer2 = RelaxedNewton(convergence_tolerance=0,
    #                           convergence_level=1,
    #                           iteration_limit=2,
    #                           callback=convergence_measure)
    #
    # minimizer1 = VL_BFGS(convergence_tolerance=0,
    #                    iteration_limit=5,
    #                    callback=convergence_measure,
    #                    max_history_length=3)



    # Setting starting position
    flat_power = Field(p_space,val=10e-8)
    m0 = flat_power.power_synthesize(real_signal=True)

    t0 = Field(p_space, val=log(1./(1+p_space.kindex)**2))
    # t0 = Field(p_space,val=-8)
    # t0 = log(sp.copy()**2)
    S0 = create_power_operator(h_space, power_spectrum=exp(t0),
                               distribution_strategy=distribution_strategy)



    for i in range(100):
        S0 = create_power_operator(h_space, power_spectrum=exp(t0),
                              distribution_strategy=distribution_strategy)

        # Initializing the  nonlinear Wiener Filter energy
        map_energy = NonlinearWienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S0)
        # Minimization with chosen minimizer
        (map_energy, convergence) = minimizer1(map_energy)
        # Updating parameters for correlation structure reconstruction
        m0 = map_energy.position
        D0 = map_energy.curvature
        # Initializing the power energy with updated parameters
        power_energy = CriticalPowerEnergy(position=t0, m=m0, D=D0, sigma=10., samples=3)
        (power_energy, convergence) = minimizer1(power_energy)
        # Setting new power spectrum
        t0 = power_energy.position
        plot_parameters(m0,t0,log(sp**2),realized_power)

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

    f_m_data = function(m).val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=f_m_data)], filename='f_map.html')
    f_ss_data = function(ss).val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=f_ss_data)], filename='f_ss.html')
