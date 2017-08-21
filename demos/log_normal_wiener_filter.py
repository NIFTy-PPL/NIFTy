# -*- coding: utf-8 -*-

from nifty import *


if __name__ == "__main__":
    nifty_configuration['default_distribution_strategy'] = 'fftw'

    # Setting up parameters    |\label{code:wf_parameters}|
    correlation_length = 1.     # Typical distance over which the field is correlated
    field_variance = 2.         # Variance of field in position space
    response_sigma = 0.02       # Smoothing length of response (in same unit as L)
    signal_to_noise = 100       # The signal to noise ratio
    np.random.seed(43)          # Fixing the random seed
    def power_spectrum(k):      # Defining the power spectrum
        a = 4 * correlation_length * field_variance**2
        return a / (1 + k * correlation_length) ** 4

    # Setting up the geometry |\label{code:wf_geometry}|
    L = 2. # Total side-length of the domain
    N_pixels = 128 # Grid resolution (pixels per axis)
    #signal_space = RGSpace([N_pixels, N_pixels], distances=L/N_pixels)
    signal_space = HPSpace(16)
    harmonic_space = FFTOperator.get_default_codomain(signal_space)
    fft = FFTOperator(harmonic_space, target=signal_space, target_dtype=np.float)
    power_space = PowerSpace(harmonic_space)

    # Creating the mock signal |\label{code:wf_mock_signal}|
    S = create_power_operator(harmonic_space, power_spectrum=power_spectrum)
    mock_power = Field(power_space, val=power_spectrum)
    mock_signal = fft(mock_power.power_synthesize(real_signal=True))

    # Setting up an exemplary response
    mask = Field(signal_space, val=1.)
    N10 = int(N_pixels/10)
    #mask.val[N10*5:N10*9, N10*5:N10*9] = 0.
    R = ResponseOperator(signal_space, sigma=(response_sigma,), exposure=(mask,)) #|\label{code:wf_response}|
    data_domain = R.target[0]
    R_harmonic = ComposedOperator([fft, R], default_spaces=[0, 0])

    # Setting up the noise covariance and drawing a random noise realization
    N = DiagonalOperator(data_domain, diagonal=mock_signal.var()/signal_to_noise, bare=True)
    noise = Field.from_random(domain=data_domain, random_type='normal',
                              std=mock_signal.std()/np.sqrt(signal_to_noise), mean=0)
    data = R(exp(mock_signal)) + noise #|\label{code:wf_mock_data}|

    # Wiener filter
    m0 = Field(harmonic_space, val=0.j)
    energy = library.LogNormalWienerFilterEnergy(m0, data, R_harmonic, N, S)


    minimizer1 = VL_BFGS(convergence_tolerance=1e-5,
                         iteration_limit=3000,
                         #callback=convergence_measure,
                         max_history_length=20)

    minimizer2 = RelaxedNewton(convergence_tolerance=1e-5,
                               iteration_limit=10,
                               #callback=convergence_measure
                               )
    minimizer3 = SteepestDescent(convergence_tolerance=1e-5, iteration_limit=1000)


#    me1 = minimizer1(energy)
#    me2 = minimizer2(energy)
#    me3 = minimizer3(energy)

#    m1 = fft(me1[0].position)
#    m2 = fft(me2[0].position)
#    m3 = fft(me3[0].position)
#


#    # Probing the variance
#    class Proby(DiagonalProberMixin, Prober): pass
#    proby = Proby(signal_space, probe_count=100)
#    proby(lambda z: fft(wiener_curvature.inverse_times(fft.inverse_times(z))))
#
#    sm = SmoothingOperator(signal_space, sigma=0.02)
#    variance = sm(proby.diagonal.weight(-1))

    #Plotting #|\label{code:wf_plotting}|
    #plotter = plotting.RG2DPlotter(color_map=plotting.colormaps.PlankCmap())
    plotter = plotting.HealpixPlotter(color_map=plotting.colormaps.PlankCmap())

    plotter.figure.xaxis = plotting.Axis(label='Pixel Index')
    plotter.figure.yaxis = plotting.Axis(label='Pixel Index')

    plotter.plot.zmax = 5; plotter.plot.zmin = -5
##    plotter(variance, path = 'variance.html')
#    #plotter.plot.zmin = exp(mock_signal.min());
#    plotter(mock_signal.real, path='mock_signal.html')
#    plotter(Field(signal_space, val=np.log(data.val.get_full_data().real).reshape(signal_space.shape)),
#            path = 'log_of_data.html')
#
#    plotter(m1.real, path='m_LBFGS.html')
#    plotter(m2.real, path='m_Newton.html')
#    plotter(m3.real, path='m_SteepestDescent.html')
#
