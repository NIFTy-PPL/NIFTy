# -*- coding: utf-8 -*-

import nifty2go as ift
from nifty2go import plotting
import numpy as np


if __name__ == "__main__":
    # Setting up parameters    |\label{code:wf_parameters}|
    correlation_length_scale = 1.  # Typical distance over which the field is correlated
    fluctuation_scale = 2.         # Variance of field in position space
    response_sigma = 0.05          # Smoothing length of response (in same unit as L)
    signal_to_noise = 1.5          # The signal to noise ratio
    np.random.seed(43)             # Fixing the random seed
    def power_spectrum(k):         # Defining the power spectrum
        a = 4 * correlation_length_scale * fluctuation_scale**2
        return a / (1 + (k * correlation_length_scale)**2) ** 2

    # Setting up the geometry |\label{code:wf_geometry}|
    L = 2.  # Total side-length of the domain
    N_pixels = 512  # Grid resolution (pixels per axis)
    signal_space = ift.RGSpace([N_pixels, N_pixels], distances=L/N_pixels)
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
    mask.val[N10*5:N10*9, N10*5:N10*9] = 0.
    R = ift.ResponseOperator(signal_space, sigma=(response_sigma,), exposure=(mask,))  #|\label{code:wf_response}|
    data_domain = R.target[0]
    R_harmonic = ift.ComposedOperator([fft, R], default_spaces=[0, 0])

    # Setting up the noise covariance and drawing a random noise realization
    N = ift.DiagonalOperator(data_domain, diagonal=mock_signal.var()/signal_to_noise, bare=True)
    noise = ift.Field.from_random(domain=data_domain, random_type='normal',
                                  std=mock_signal.std()/np.sqrt(signal_to_noise), mean=0)
    data = R(mock_signal) + noise  #|\label{code:wf_mock_data}|

    # Wiener filter
    j = R_harmonic.adjoint_times(N.inverse_times(data))
    ctrl = ift.DefaultIterationController(verbose=False,tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=ctrl,preconditioner=S.times)
    wiener_curvature = ift.library.WienerFilterCurvature(S=S, N=N, R=R_harmonic,inverter=inverter)
    m_k = wiener_curvature.inverse_times(j)  #|\label{code:wf_wiener_filter}|
    m = fft(m_k)

    # Probing the uncertainty |\label{code:wf_uncertainty_probing}|
    class Proby(ift.DiagonalProberMixin, ift.Prober): pass
    proby = Proby(signal_space, probe_count=10)
#    class Proby(ift.DiagonalProberMixin, ift.ParallelProber): pass
#    proby = Proby(signal_space, probe_count=10,ncpu=2)
    proby(lambda z: fft(wiener_curvature.inverse_times(fft.inverse_times(z))))  #|\label{code:wf_variance_fft_wrap}|

    sm = ift.FFTSmoothingOperator(signal_space, sigma=0.03)
    variance = ift.sqrt(sm(proby.diagonal.weight(-1)))  #|\label{code:wf_variance_weighting}|

    # Plotting #|\label{code:wf_plotting}|
    plotter = plotting.RG2DPlotter(color_map=plotting.colormaps.PlankCmap())
    plotter.figure.xaxis = ift.plotting.Axis(label='Pixel Index')
    plotter.figure.yaxis = ift.plotting.Axis(label='Pixel Index')
    plotter.plot.zmax = variance.max(); plotter.plot.zmin = 0
    plotter(variance, path = 'uncertainty.html')
    plotter.plot.zmax = mock_signal.max(); plotter.plot.zmin = mock_signal.min()
    plotter(mock_signal, path='mock_signal.html')
    plotter(ift.Field(signal_space, val=data.val), path='data.html')
    plotter(m, path='map.html')
