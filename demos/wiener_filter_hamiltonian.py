
from nifty import *

import plotly.offline as pl
import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(42)

class AdjointFFTResponse(LinearOperator):
    def __init__(self, FFT, R, default_spaces=None):
        super(ResponseOperator, self).__init__(default_spaces)
        self._domain = FFT.target
        self.target = R.target
        self.R = R
        self.FFT = FFT

    def _times(self, x):
        return self.R(self.FFT.adjoint_times(x))

    def _adjoint_times(self, x):
        return self.FFT(self.R.adjoint_times(x))



if __name__ == "__main__":

    distribution_strategy = 'not'

    # Set up spaces and fft transformation
    s_space = RGSpace([512, 512])
    fft = FFTOperator(s_space)
    h_space = fft.target[0]
    p_space = PowerSpace(h_space, distribution_strategy=distribution_strategy)

    # create the field instances and power operator
    pow_spec = (lambda k: (42 / (k + 1) ** 3))
    S = create_power_operator(h_space, power_spectrum=pow_spec,
                              distribution_strategy=distribution_strategy)

    sp = Field(p_space, val=lambda z: pow_spec(z)**(1./2),
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)
    ss = fft.inverse_times(sh)

    # model the measurement process
    Instrument = SmoothingOperator(s_space, sigma=0.01)

#    Instrument = DiagonalOperator(s_space, diagonal=1.)
#    Instrument._diagonal.val[200:400, 200:400] = 0
    R = AdjointFFTResponse(fft, Instrument)
    signal_to_noise = 1
    N = DiagonalOperator(s_space, diagonal=ss.var()/signal_to_noise, bare=True)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=ss.std()/np.sqrt(signal_to_noise),
                          mean=0)

    # create mock data
    d = R(sh) + n

    def distance_measure(energy, iteration):
        x = energy.value
        print (x, iteration)

#    minimizer = SteepestDescent(convergence_tolerance=0,
#                                iteration_limit=50,
#                                callback=distance_measure)

    minimizer = RelaxedNewton(convergence_tolerance=0,
                              iteration_limit=2,
                              callback=distance_measure)

#    minimizer = VL_BFGS(convergence_tolerance=0,
#                        iteration_limit=50,
#                        callback=distance_measure,
#                        max_history_length=3)

    solution = energy.analytic_solution()
    m0 = Field(s_space, val=1.)

    energy = WienerFilterEnergy(position=m0, D=D, j=j)

    (energy, convergence) = minimizer(energy)

    m = fft.adjoint_times(energy.position)

    d_data = d.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=d_data)], filename='data.html')


    ss_data = ss.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=ss_data)], filename='ss.html')

    sh_data = sh.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=sh_data)], filename='sh.html')

    j_data = j.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=j_data)], filename='j.html')

    jabs_data = np.abs(j.val.get_full_data())
    jphase_data = np.angle(j.val.get_full_data())
    if rank == 0:
        pl.plot([go.Heatmap(z=jabs_data)], filename='j_abs.html')
        pl.plot([go.Heatmap(z=jphase_data)], filename='j_phase.html')

    m_data = m.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=m_data)], filename='map.html')

#    grad_data = grad.val.get_full_data().real
#    if rank == 0:
#        pl.plot([go.Heatmap(z=grad_data)], filename='grad.html')
