
import d2o

from nifty import *

import plotly.offline as pl
import plotly.graph_objs as go
from nifty.library.wiener_filter import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

# d2o.random.seed(42)


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

    nifty_configuration['default_distribution_strategy'] = 'not'
    nifty_configuration['harmonic_rg_base'] = 'real'

    # Set up position space
    s_space = RGSpace([128, 128])
    # s_space = HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    fft = FFTOperator(s_space)
    h_space = fft.target[0]

    # Setting up power space
    p_space = PowerSpace(h_space)

    # Choosing the prior correlation structure and defining
    # correlation operator
    p_spec = (lambda k: (42 / (k + 1) ** 3))

    S = create_power_operator(h_space, power_spectrum=p_spec)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = Field(p_space, val=p_spec)
    sh = sp.power_synthesize(real_signal=True)
    ss = fft.adjoint_times(sh)

    # Choosing the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.05)
    Instrument = DiagonalOperator(s_space, diagonal=1.)
    Instrument._diagonal.val[64:80, 32:80] = 0.

    # Adding a harmonic transformation to the instrument
    R = AdjointFFTResponse(fft, Instrument)
    signal_to_noise = 1.
    ndiag = Field(s_space, val=ss.var()/signal_to_noise).weight()
    N = DiagonalOperator(s_space, diagonal=ndiag)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=ss.std()/np.sqrt(signal_to_noise),
                          mean=0)

    # Creating the mock data
    d = R(sh) + n
    j = R.adjoint_times(N.inverse_times(d))

    # Choosing the minimization strategy

    def convergence_measure(energy, iteration):  # returns current energy
        x = energy.value
        print(x, iteration)

#    minimizer = SteepestDescent(convergence_tolerance=0,
#                                iteration_limit=50,
#                                callback=convergence_measure)



    controller = GradientNormController(iteration_limit=50,
                                        callback=convergence_measure)
    minimizer = VL_BFGS(controller=controller)

    controller = GradientNormController(iteration_limit=1,
                                        callback=convergence_measure)
    minimizer = RelaxedNewton(controller=controller)


    #
    # minimizer = VL_BFGS(convergence_tolerance=0,
    #                    iteration_limit=50,
    #                    callback=convergence_measure,
    #                    max_history_length=3)
    #

    # Setting starting position
    m0 = Field(h_space, val=.0)

    # Initializing the Wiener Filter energy
    energy = WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S)
    D0 = energy.curvature

    # Solving the problem analytically
    m0 = D0.inverse_times(j)
    energy, convergence = minimizer(energy)
    m = energy.position
    D = energy.curvature
    m = minimizer(energy)[0].position

    plotter = plotting.RG2DPlotter()
    plotter(ss, path='signal.html')
    plotter(fft.inverse_times(m), path='m.html')


    sample_variance = Field(s_space, val=0.)
    sample_mean = Field(s_space, val=0.)

    # sampling the uncertainty map
    n_samples = 50
    for i in range(n_samples):
        sample = fft.adjoint_times(sugar.generate_posterior_sample(m, D))
        sample_variance += sample**2
        sample_mean += sample
    sample_mean /= n_samples
    sample_variance /= n_samples
    variance = (sample_variance - sample_mean**2)
