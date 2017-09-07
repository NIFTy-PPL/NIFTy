# -*- coding: utf-8 -*-

import nifty2go as ift
import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
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
    signal_space = ift.HPSpace(16)
    harmonic_space = signal_space.get_default_codomain()
    fft = ift.FFTOperator(harmonic_space, target=signal_space)
    power_space = ift.PowerSpace(harmonic_space)

    # Creating the mock signal |\label{code:wf_mock_signal}|
    S = ift.create_power_operator(harmonic_space, power_spectrum=power_spectrum)
    mock_power = ift.Field(power_space, val=power_spectrum(power_space.kindex))
    mock_signal = fft(mock_power.power_synthesize(real_signal=True))

    # Setting up an exemplary response
    mask = ift.Field(signal_space, val=1.)
    N10 = int(N_pixels/10)
    #mask.val[N10*5:N10*9, N10*5:N10*9] = 0.
    R = ift.ResponseOperator(signal_space, sigma=(response_sigma,), exposure=(mask,)) #|\label{code:wf_response}|
    data_domain = R.target[0]
    R_harmonic = ift.ComposedOperator([fft, R], default_spaces=[0, 0])

    # Setting up the noise covariance and drawing a random noise realization
    N = ift.DiagonalOperator(data_domain, diagonal=mock_signal.var()/signal_to_noise, bare=True)
    noise = ift.Field.from_random(domain=data_domain, random_type='normal',
                              std=mock_signal.std()/np.sqrt(signal_to_noise), mean=0)
    data = R(ift.exp(mock_signal)) + noise #|\label{code:wf_mock_data}|

    # Wiener filter
    m0 = ift.Field(harmonic_space, val=0.)
    ctrl = ift.DefaultIterationController(verbose=False,tol_abs_gradnorm=1)
    ctrl2 = ift.DefaultIterationController(verbose=True,tol_abs_gradnorm=0.1, name="outer")
    inverter = ift.ConjugateGradient(controller=ctrl)
    energy = ift.library.LogNormalWienerFilterEnergy(m0, data, R_harmonic, N, S, inverter=inverter)
    minimizer1 = ift.VL_BFGS(controller=ctrl2,max_history_length=20)
    minimizer2 = ift.RelaxedNewton(controller=ctrl2)
    minimizer3 = ift.SteepestDescent(controller=ctrl2)

    print type(energy.value)
    me1 = minimizer1(energy)
    me2 = minimizer2(energy)
    me3 = minimizer3(energy)

    m1 = fft(me1[0].position)
    m2 = fft(me2[0].position)
    m3 = fft(me3[0].position)


    # Probing the variance
    #class Proby(ift.DiagonalProberMixin, ift.Prober): pass
    #proby = Proby(signal_space, probe_count=100)
    #proby(lambda z: fft(wiener_curvature.inverse_times(fft.inverse_times(z))))

    #sm = SmoothingOperator(signal_space, sigma=0.02)
    #variance = sm(proby.diagonal.weight(-1))

    #Plotting #|\label{code:wf_plotting}|
    #plotter = plotting.RG2DPlotter(color_map=plotting.colormaps.PlankCmap())
    plotter = ift.plotting.HealpixPlotter(color_map=ift.plotting.colormaps.PlankCmap())

    plotter.figure.xaxis = ift.plotting.Axis(label='Pixel Index')
    plotter.figure.yaxis = ift.plotting.Axis(label='Pixel Index')

    plotter.plot.zmax = 5; plotter.plot.zmin = -5
    #plotter(variance, path = 'variance.html')
#    #plotter.plot.zmin = exp(mock_signal.min());
#    plotter(mock_signal.real, path='mock_signal.html')
#    plotter(Field(signal_space, val=np.log(data.val.get_full_data().real).reshape(signal_space.shape)),
#            path = 'log_of_data.html')
#
    plotter(m1.real, path='m_LBFGS.html')
    plotter(m2.real, path='m_Newton.html')
    plotter(m3.real, path='m_SteepestDescent.html')
#
