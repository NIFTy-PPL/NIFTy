
from nifty import *
import plotly.offline as pl
import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank


if __name__ == "__main__":

    distribution_strategy = 'fftw'

    s_space = RGSpace([512, 512], dtype=np.float)
    fft = FFTOperator(s_space)
    h_space = fft.target[0]
    p_space = PowerSpace(h_space, distribution_strategy=distribution_strategy)

    pow_spec = (lambda k: (42 / (k + 1) ** 3))

    S = create_power_operator(h_space, power_spectrum=pow_spec,
                              distribution_strategy=distribution_strategy)

    sp = Field(p_space, val=lambda z: pow_spec(z)**(1./2),
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)
    ss = fft.inverse_times(sh)

    R = SmoothingOperator(s_space, sigma=0.01)

#    R = DiagonalOperator(s_space, diagonal=1.)
#    R._diagonal.val[200:400, 200:400] = 0

    signal_to_noise = 1
    N = DiagonalOperator(s_space, diagonal=ss.var()/signal_to_noise, bare=True)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=ss.std()/np.sqrt(signal_to_noise),
                          mean=0)
    #n.val.data.imag[:] = 0

    d = R(ss) + n
    j = R.adjoint_times(N.inverse_times(d))
    D = PropagatorOperator(S=S, N=N, R=R)

    def energy(x):
        DIx = D.inverse_times(x)
        H = 0.5 * DIx.dot(x) - j.dot(x)
        return H.real

    def gradient(x):
        DIx = D.inverse_times(x)
        g = DIx - j
        return_g = g.copy_empty(dtype=np.float)
        return_g.val = g.val.real
        return return_g

    def distance_measure(x, fgrad, iteration):
        print (iteration, ((x-ss).norm()/ss.norm()).real)

    minimizer = SteepestDescent(convergence_tolerance=0,
                                iteration_limit=50,
                                callback=distance_measure)
    minimizer = VL_BFGS(convergence_tolerance=0,
                        iteration_limit=50,
                        callback=distance_measure,
                        max_history_length=5)


    m0 = Field(s_space, val=1)

    (m, convergence) = minimizer(m0, energy, gradient)


    grad = gradient(m)

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

    grad_data = grad.val.get_full_data().real
    if rank == 0:
        pl.plot([go.Heatmap(z=grad_data)], filename='grad.html')
