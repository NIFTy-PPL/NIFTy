
from nifty import *

import plotly.offline as pl
import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(42)

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
    p_space = PowerSpace(h_space, distribution_strategy=distribution_strategy)

    # Choosing the prior correlation structure and defining correlation operator
    p_spec = (lambda k: (42 / (k + 1) ** 3))

    S = create_power_operator(h_space, power_spectrum=p_spec,
                              distribution_strategy=distribution_strategy)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = Field(p_space, val=p_spec,
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)
    ss = fft.adjoint_times(sh)

    # Choosing the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.05)
    Instrument = DiagonalOperator(s_space, diagonal=1.)
#    Instrument._diagonal.val[200:400, 200:400] = 0

    #Adding a harmonic transformation to the instrument
    R = AdjointFFTResponse(fft, Instrument)
    signal_to_noise = 1.
    N = DiagonalOperator(s_space, diagonal=ss.var()/signal_to_noise, bare=True)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=ss.std()/np.sqrt(signal_to_noise),
                          mean=0)

    # Creating the mock data
    d = R(sh) + n
    j = R.adjoint_times(N.inverse_times(d))

    # Choosing the minimization strategy

    def convergence_measure(energy, iteration): # returns current energy
        x = energy.value
        print (x, iteration)

#    minimizer = SteepestDescent(convergence_tolerance=0,
#                                iteration_limit=50,
#                                callback=convergence_measure)

    minimizer = RelaxedNewton(convergence_tolerance=0,
                              iteration_limit=1,
                              callback=convergence_measure)
    #
    # minimizer = VL_BFGS(convergence_tolerance=0,
    #                    iteration_limit=50,
    #                    callback=convergence_measure,
    #                    max_history_length=3)
    #

    inverter = ConjugateGradient(convergence_level=3,
                                 convergence_tolerance=10e-5,
                                 preconditioner=None)
    # Setting starting position
    m0 = Field(h_space, val=.0)

    # Initializing the Wiener Filter energy
    energy = WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S, inverter=inverter)
    D0 = energy.curvature

    # Solving the problem analytically
    m0 = D0.inverse_times(j)

    sample_variance = Field(sh.power_analyze(logarithmic=False).domain,val=0. + 0j)
    sample_mean = Field(sh.domain,val=0. + 0j)

    # sampling the uncertainty map
    n_samples = 1
    for i in range(n_samples):
        sample = sugar.generate_posterior_sample(m0,D0)
        sample_variance += sample**2
        sample_mean += sample
    variance = sample_variance/n_samples - (sample_mean/n_samples)

