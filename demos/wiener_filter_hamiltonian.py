from __future__ import division
from __future__ import print_function
from builtins import object

from nifty import *

import plotly.offline as pl
import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

np.random.seed(42)

class WienerFilterEnergy(Energy):
    def __init__(self, position, D, j):
        # in principle not necessary, but useful in order to make the signature
        # explicit
        super(WienerFilterEnergy, self).__init__(position)
        self.D = D
        self.j = j

    def at(self, position):
        return self.__class__(position, D=self.D, j=self.j)

    @property
    def value(self):
        D_inv_x = self.D_inverse_x()
        H = 0.5 * D_inv_x.vdot(self.position) - self.j.dot(self.position)
        return H.real

    @property
    def gradient(self):
        D_inv_x = self.D_inverse_x()
        g = D_inv_x - self.j
        return_g = g.copy_empty(dtype=np.float)
        return_g.val = g.val.real
        return return_g

    @property
    def curvature(self):
        class Dummy(object):
            def __init__(self, x):
                self.x = x
            def inverse_times(self, *args, **kwargs):
                return self.x.times(*args, **kwargs)
        my_dummy = Dummy(self.D)
        return my_dummy


    @memo
    def D_inverse_x(self):
        return D.inverse_times(self.position)


if __name__ == "__main__":

    distribution_strategy = 'not'

    # Set up spaces and fft transformation
    s_space = RGSpace([512, 512])
    fft = FFTOperator(s_space)
    h_space = fft.target[0]
    p_space = PowerSpace(h_space, distribution_strategy=distribution_strategy)

    # create the field instances and power operator
    pow_spec = (lambda k: (42. / (k + 1) ** 3))
    S = create_power_operator(h_space, power_spectrum=pow_spec,
                              distribution_strategy=distribution_strategy)

    sp = Field(p_space, val=lambda z: pow_spec(z)**0.5,
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)
    ss = fft.inverse_times(sh)

    # model the measurement process
    R = SmoothingOperator(s_space, sigma=0.01)
#    R = DiagonalOperator(s_space, diagonal=1.)
#    R._diagonal.val[200:400, 200:400] = 0

    signal_to_noise = 1
    N = DiagonalOperator(s_space, diagonal=ss.var()/signal_to_noise, bare=True)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=ss.std()/np.sqrt(signal_to_noise),
                          mean=0)

    # create mock data
    d = R(ss) + n

    # set up reconstruction objects
    j = R.adjoint_times(N.inverse_times(d))
    D = PropagatorOperator(S=S, N=N, R=R)

    def distance_measure(energy, iteration):
        x = energy.position
        print((iteration, (x-ss).norm()/ss.norm()).real))

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

    m0 = Field(s_space, val=1.)

    energy = WienerFilterEnergy(position=m0, D=D, j=j)

    (energy, convergence) = minimizer(energy)

    m = energy.position

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
