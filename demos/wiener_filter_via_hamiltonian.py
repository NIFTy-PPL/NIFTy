from nifty import *

import plotly.offline as pl
import plotly.graph_objs as go
from nifty.library.wiener_filter import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(42)

# class AdjointFFTResponse(LinearOperator):
#     def __init__(self, FFT, R, default_spaces=None):
#         super(AdjointFFTResponse, self).__init__(default_spaces)
#         self._domain = FFT.target
#         self._target = R.target
#         self.R = R
#         self.FFT = FFT
#
#     def _times(self, x, spaces=None):
#         return self.R(self.FFT.adjoint_times(x))
#
#     def _adjoint_times(self, x, spaces=None):
#         return self.FFT(self.R.adjoint_times(x))
#     @property
#     def domain(self):
#         return self._domain
#
#     @property
#     def target(self):
#         return self._target
#
#     @property
#     def unitary(self):
#         return False
#


if __name__ == "__main__":

    distribution_strategy = 'equal'

    # Set up position space
    signal_space = RGSpace([256,256])
    # s_space = HPSpace(32)
    harmonic_space = FFTOperator.get_default_codomain(signal_space)

    fft = FFTOperator(harmonic_space, target=signal_space,
                      domain_dtype=np.complex, target_dtype=np.float)
    # Define harmonic transformation and associated harmonic space

    # Setting up power space
    power_space = PowerSpace(harmonic_space,
                             distribution_strategy=distribution_strategy)

    # Choosing the prior correlation structure and defining correlation operator
    power_spectrum = (lambda k: (42 / (k + 1) ** 3))

    S = create_power_operator(harmonic_space, power_spectrum=power_spectrum,
                              distribution_strategy=distribution_strategy)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = Field(power_space, val=power_spectrum,
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)
    ss = fft(sh)
    # Choosing the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.05)
    mask = Field(signal_space, val=1,
                 distribution_strategy=distribution_strategy)
    mask.val[63:127, 63:127] = 0.
    Instrument = DiagonalOperator(signal_space, diagonal=mask)

    #Adding a harmonic transformation to the instrument
    R = ComposedOperator([fft, Instrument], default_spaces=[0, 0])
    signal_to_noise = 1.
    N = DiagonalOperator(signal_space, diagonal=ss.var()/signal_to_noise,
                         bare=True,
                         distribution_strategy=distribution_strategy)
    n = Field.from_random(domain=signal_space,
                          random_type='normal',
                          std=ss.std()/np.sqrt(signal_to_noise),
                          mean=0, distribution_strategy=distribution_strategy)

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

    minimizer = VL_BFGS(convergence_tolerance=0,
                       iteration_limit=500,
                       callback=convergence_measure,
                       max_history_length=3)


    inverter = ConjugateGradient(convergence_level=3,
                                 convergence_tolerance=1e-5,
                                 preconditioner=None)
    # Setting starting position
    m0 = Field(harmonic_space, val=.0,
               distribution_strategy=distribution_strategy)

    # Initializing the Wiener Filter energy
    energy = WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S,
                                inverter=inverter)
    D0 = energy.curvature

    # Solving the problem analytically
    m0 = D0.inverse_times(j)
    # solution, convergence = minimizer(energy)
    # m0 = solution.position
    m0_s = Field(signal_space,val=fft(m0).val.get_full_data().real)

    plotter = plotting.RG2DPlotter()
    plotter.title = 'mock_signal.html';
    plotter(ss)
    plotter.title = 'data.html'
    plotter(Field(signal_space,
                  val=Instrument.adjoint_times(d).val.get_full_data()\
                  .reshape(signal_space.shape)))
    plotter.title = 'map.html'; plotter(m0_s)
    #
    # sample_variance = Field(sh.domain,val=0. + 0j,
    #                         distribution_strategy=distribution_strategy)
    # sample_mean = Field(sh.domain,val=0. + 0j,
    #                         distribution_strategy=distribution_strategy)
    #
    # # sampling the uncertainty map
    # n_samples = 1
    # for i in range(n_samples):
    #     sample = sugar.generate_posterior_sample(m0,D0)
    #     sample_variance += sample**2
    #     sample_mean += sample
    # variance = sample_variance/n_samples - (sample_mean/n_samples)**2

