
from nifty import *
#import plotly.offline as pl
#import plotly.graph_objs as go

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank


if __name__ == "__main__":

    distribution_strategy = 'not'
    
    #Setting up physical constants
    #total length of Interval or Volume the field lives on, e.g. in meters
    L = 2.
    #typical distance over which the field is correlated (in same unit as L)
    correlation_length = 0.1
    #variance of field in position space sqrt(<|s_x|^2>) (in unit of s)
    field_variance = 2.
    #smoothing length of response (in same unit as L)
    response_sigma = 0.1
    
    #defining resolution (pixels per dimension)
    N_pixels = 512
    
    #Setting up derived constants
    k_0 = 1./correlation_length
    #note that field_variance**2 = a*k_0/4. for this analytic form of power
    #spectrum
    a = field_variance**2/k_0*4.
    pow_spec = (lambda k: a / (1 + k/k_0) ** 4)
    pixel_width = L/N_pixels
    
    # Setting up the geometry
    s_space = RGSpace([N_pixels, N_pixels], distances = pixel_width)
    fft = FFTOperator(s_space)
    h_space = fft.target[0]
    p_space = PowerSpace(h_space, distribution_strategy=distribution_strategy)


    # Creating the mock data

    S = create_power_operator(h_space, power_spectrum=pow_spec,
                              distribution_strategy=distribution_strategy)

    sp = Field(p_space, val=pow_spec,
               distribution_strategy=distribution_strategy)
    sh = sp.power_synthesize(real_signal=True)
    ss = fft.inverse_times(sh)

    R = SmoothingOperator(s_space, sigma=response_sigma)

    signal_to_noise = 1
    N = DiagonalOperator(s_space, diagonal=ss.var()/signal_to_noise, bare=True)
    n = Field.from_random(domain=s_space,
                          random_type='normal',
                          std=ss.std()/np.sqrt(signal_to_noise),
                          mean=0)

    d = R(ss) + n

    # Wiener filter

    j = R.adjoint_times(N.inverse_times(d))
    D = PropagatorOperator(S=S, N=N, R=R)

    m = D(j)

    d_data = d.val.get_full_data().real
    m_data = m.val.get_full_data().real
    ss_data = ss.val.get_full_data().real
#    if rank == 0:
#        pl.plot([go.Heatmap(z=d_data)], filename='data.html')
#        pl.plot([go.Heatmap(z=m_data)], filename='map.html')
#        pl.plot([go.Heatmap(z=ss_data)], filename='map_orig.html')
