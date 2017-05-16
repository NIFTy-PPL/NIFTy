
from nifty import *

import plotly.offline as pl
import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(42)

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



if __name__ == "__main__":

    distribution_strategy = 'not'

    # Set up position space
    s_space = RGSpace([128,128])
    # s_space = HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    fft = FFTOperator(s_space)
    h_space = fft.target[0]

    # Setting up power space
    p_space = PowerSpace(h_space, distribution_strategy=distribution_strategy)

    # Choosing the prior correlation structure and defining correlation operator
    pow_spec = (lambda k: (0.42 / (k + 1) ** 3))
    S = create_power_operator(h_space, power_spectrum=pow_spec,
                              distribution_strategy=distribution_strategy)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = Field(p_space, val=lambda z: pow_spec(z)**(1./2),
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)


    # Choosing the measurement instrument
#    Instrument = SmoothingOperator(s_space, sigma=0.01)
    Instrument = DiagonalOperator(s_space, diagonal=1.)
#    Instrument._diagonal.val[200:400, 200:400] = 0

    #   Choosing nonlinearity

    # function = exp
    # derivative = exp
    def function(x):
        return 0.5 * tanh(x) + 0.5
    def derivative(x):
        return 0.5*(1 - tanh(x)**2)
    # def function(x):
    #     return x
    # def derivative(x):
    #     return 1
    #
    # def function(x):
    #     return 0.5*x**2 + x
    # def derivative(x):
    #     return x + 1
    #
    # def function(x):
    #     return 0.9*x**4 +0.2*x**2 + x
    # def derivative(x):
    #     return 0.9*4*x**3 + 0.4*x +1
    #

    #Adding a harmonic transformation to the instrument
    R = NonlinearResponse(fft, Instrument, function, derivative)
    noise = 0.01
    N = DiagonalOperator(s_space, diagonal=noise, bare=True)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=sqrt(noise),
                          mean=0)

    # Creating the mock data
    d = R(sh) + n

    # Choosing the minimization strategy

    def convergence_measure(energy, iteration): # returns current energy
        x = energy.value
        print (x, iteration)

#    minimizer = SteepestDescent(convergence_tolerance=0,
#                                iteration_limit=50,
#                                callback=convergence_measure)
#
    minimizer = RelaxedNewton(convergence_tolerance=0,
                              convergence_level=1,
                              iteration_limit=10,
                              callback=convergence_measure)

    # minimizer = VL_BFGS(convergence_tolerance=0,
    #                    iteration_limit=50,
    #                    callback=convergence_measure,
    #                    max_history_length=3)
    #
    #

    # Setting starting position
    m0 = Field.from_random(random_type="normal",domain = h_space, std=0.001)

    # Initializing the Wiener Filter energy
    energy = NonlinearWienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S)

    # Solving the problem with chosen minimization strategy
    (energy, convergence) = minimizer(energy)

    # Transforming fields to position space for plotting

    ss = fft.adjoint_times(sh)
    m = fft.adjoint_times(energy.position)


    # Plotting

    d_data = d.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=d_data)], filename='data.html')


    ss_data = ss.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=ss_data)], filename='ss.html')

    sh_data = sh.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=sh_data)], filename='sh.html')


    m_data = m.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=m_data)], filename='map.html')

