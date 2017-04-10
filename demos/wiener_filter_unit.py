from nifty import *
from mpi4py import MPI
import plotly.offline as py
import plotly.graph_objs as go

comm = MPI.COMM_WORLD
rank = comm.rank




def plot_maps(x, name):

    trace = [None]*len(x)

    keys = x.keys()
    field = x[keys[0]]
    domain = field.domain[0]
    shape = len(domain.shape)
    max_n = domain.shape[0]*domain.distances[0]
    step = domain.distances[0]
    x_axis = np.arange(0, max_n, step)

    if shape == 1:
        for ii in xrange(len(x)):
            trace[ii] = go.Scatter(x= x_axis, y=x[keys[ii]].val.get_full_data(), name=keys[ii])
        fig = go.Figure(data=trace)

        py.plot(fig, filename=name)

    elif shape == 2:
        for ii in xrange(len(x)):
            py.plot([go.Heatmap(z=x[keys[ii]].val.get_full_data())], filename=keys[ii])
    else:
        raise TypeError("Only 1D and 2D field plots are supported")

def plot_power(x, name):

    layout = go.Layout(
        xaxis=dict(
            type='log',
            autorange=True
        ),
        yaxis=dict(
            type='log',
            autorange=True
        )
    )

    trace = [None]*len(x)

    keys = x.keys()
    field = x[keys[0]]
    domain = field.domain[0]
    x_axis = domain.kindex

    for ii in xrange(len(x)):
        trace[ii] = go.Scatter(x= x_axis, y=x[keys[ii]].val.get_full_data(), name=keys[ii])

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=name)

np.random.seed(42)

if __name__ == "__main__":

    distribution_strategy = 'not'

    # setting spaces
    npix = np.array([500])  # number of pixels
    total_volume = 1.  # total length

    # setting signal parameters
    lambda_s = .05  # signal correlation length
    sigma_s = 10.  # signal variance


    #setting response operator parameters
    length_convolution = .025
    exposure = 2000.



    # calculating parameters
    k_0 = 4. / (2 * np.pi * lambda_s)
    a_s = sigma_s ** 2. * lambda_s * total_volume

    # creation of spaces
    x1 = RGSpace(npix, dtype=np.float64, distances=total_volume / npix,
                 zerocenter=False)
    k1 = RGRGTransformation.get_codomain(x1)
    p1 = PowerSpace(harmonic_domain=k1, log=False, dtype=np.float64)

    # creating Power Operator with given spectrum
    spec = (lambda k: a_s / (1 + (k / k_0) ** 2) ** 2)
    p_field = Field(p1, val=spec)
    S_op = create_power_operator(k1, spec)

    # creating FFT-Operator and Response-Operator with Gaussian convolution
    Fft_op = FFTOperator(domain=x1, target=k1,
                        domain_dtype=np.float64,
                        target_dtype=np.complex128)
    R_op = ResponseOperator(x1, sigma=length_convolution,
                            exposure=exposure)

    # drawing a random field
    sk = p_field.power_synthesize(real_signal=True, mean=0.)
    s = Fft_op.inverse_times(sk)

    signal_to_noise = 1
    N_op = DiagonalOperator(R_op.target, diagonal=s.var()/signal_to_noise, bare=True)
    n = Field.from_random(domain=R_op.target,
                          random_type='normal',
                          std=s.std()/np.sqrt(signal_to_noise),
                          mean=0)

    d = R_op(s) + n

    # Wiener filter
    j = R_op.adjoint_times(N_op.inverse_times(d))
    D = PropagatorOperator(S=S_op, N=N_op, R=R_op)

    m = D(j)

    z={}
    z["signal"] = s
    z["reconstructed_map"] = m
    z["data"] = d
    z["lambda"] = R_op(s)

    plot_maps(z, "Wiener_filter.html")


