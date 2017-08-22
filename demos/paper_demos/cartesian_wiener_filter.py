# -*- coding: utf-8 -*-

import numpy as np
import nifty as ift
from nifty import plotting

from keepers import Repository

if __name__ == "__main__":
    ift.nifty_configuration['default_distribution_strategy'] = 'fftw'

    signal_to_noise = 1.5 # The signal to noise ratio


    # Setting up parameters    |\label{code:wf_parameters}|
    correlation_length_1 = 1. # Typical distance over which the field is correlated
    field_variance_1 = 2. # Variance of field in position space

    response_sigma_1 = 0.05 # Smoothing length of response (in same unit as L)

    def power_spectrum_1(k): # note: field_variance**2 = a*k_0/4.
        a = 4 * correlation_length_1 * field_variance_1**2
        return a / (1 + k * correlation_length_1) ** 4.

    # Setting up the geometry |\label{code:wf_geometry}|
    L_1 = 2. # Total side-length of the domain
    N_pixels_1 = 512 # Grid resolution (pixels per axis)

    signal_space_1 = ift.RGSpace([N_pixels_1], distances=L_1/N_pixels_1)
    harmonic_space_1 = ift.FFTOperator.get_default_codomain(signal_space_1)
    fft_1 = ift.FFTOperator(harmonic_space_1, target=signal_space_1,
                            domain_dtype=np.complex, target_dtype=np.complex)
    power_space_1 = ift.PowerSpace(harmonic_space_1)

    mock_power_1 = ift.Field(power_space_1, val=power_spectrum_1,
                             distribution_strategy='not')



    # Setting up parameters    |\label{code:wf_parameters}|
    correlation_length_2 = 1. # Typical distance over which the field is correlated
    field_variance_2 = 2. # Variance of field in position space

    response_sigma_2 = 0.01 # Smoothing length of response (in same unit as L)

    def power_spectrum_2(k): # note: field_variance**2 = a*k_0/4.
        a = 4 * correlation_length_2 * field_variance_2**2
        return a / (1 + k * correlation_length_2) ** 2.5

    # Setting up the geometry |\label{code:wf_geometry}|
    L_2 = 2. # Total side-length of the domain
    N_pixels_2 = 512 # Grid resolution (pixels per axis)

    signal_space_2 = ift.RGSpace([N_pixels_2], distances=L_2/N_pixels_2)
    harmonic_space_2 = ift.FFTOperator.get_default_codomain(signal_space_2)
    fft_2 = ift.FFTOperator(harmonic_space_2, target=signal_space_2,
                            domain_dtype=np.complex, target_dtype=np.complex)
    power_space_2 = ift.PowerSpace(harmonic_space_2, distribution_strategy='not')

    mock_power_2 = ift.Field(power_space_2, val=power_spectrum_2,
                         distribution_strategy='not')

    fft = ift.ComposedOperator((fft_1, fft_2))

    mock_power = ift.Field(domain=(power_space_1, power_space_2),
                           val=np.outer(mock_power_1.val.get_full_data(),
                                        mock_power_2.val.get_full_data()),
                                        distribution_strategy='not')

    diagonal = mock_power.power_synthesize(spaces=(0, 1), mean=1, std=0,
                                           real_signal=False)**2

    S = ift.DiagonalOperator(domain=(harmonic_space_1, harmonic_space_2),
                             diagonal=diagonal)


    np.random.seed(10)
    mock_signal = fft(mock_power.power_synthesize(real_signal=True))

    # Setting up a exemplary response
    N1_10 = int(N_pixels_1/10)
    mask_1 = ift.Field(signal_space_1, val=1.)
    mask_1.val[N1_10*7:N1_10*9] = 0.

    N2_10 = int(N_pixels_2/10)
    mask_2 = ift.Field(signal_space_2, val=1., distribution_strategy='not')
    mask_2.val[N2_10*7:N2_10*9] = 0.

    R = ift.ResponseOperator((signal_space_1, signal_space_2),
                             sigma=(response_sigma_1, response_sigma_2),
                             exposure=(mask_1, mask_2)) #|\label{code:wf_response}|
    data_domain = R.target
    R_harmonic = ift.ComposedOperator([fft, R], default_spaces=(0, 1, 0, 1))

    # Setting up the noise covariance and drawing a random noise realization
    N = ift.DiagonalOperator(data_domain, diagonal=mock_signal.var()/signal_to_noise,
                             bare=True)
    noise = ift.Field.from_random(domain=data_domain, random_type='normal',
                                  std=mock_signal.std()/np.sqrt(signal_to_noise),
                                  mean=0)
    data = R(mock_signal) + noise #|\label{code:wf_mock_data}|

    # Wiener filter
    j = R_harmonic.adjoint_times(N.inverse_times(data))
    wiener_curvature = ift.library.WienerFilterCurvature(S=S, N=N, R=R_harmonic)
    wiener_curvature._InvertibleOperatorMixin__inverter.convergence_tolerance = 1e-3

    m_k = wiener_curvature.inverse_times(j) #|\label{code:wf_wiener_filter}|
    m = fft(m_k)

    # Probing the variance
    class Proby(ift.DiagonalProberMixin, ift.Prober): pass
    proby = Proby((signal_space_1, signal_space_2), probe_count=100)
    proby(lambda z: fft(wiener_curvature.inverse_times(fft.inverse_times(z))))
#    sm = SmoothingOperator(signal_space, sigma=0.02)
#    variance = sm(proby.diagonal.weight(-1))
    variance = proby.diagonal.weight(-1)

    repo = Repository('repo_100.h5')
    repo.add(mock_signal, 'mock_signal')
    repo.add(data, 'data')
    repo.add(m, 'm')
    repo.add(variance, 'variance')
    repo.commit()

    plot_space = ift.RGSpace((N_pixels_1, N_pixels_2))
    plotter = plotting.RG2DPlotter(color_map=plotting.colormaps.PlankCmap())
    plotter.figure.xaxis = ift.plotting.Axis(label='Pixel Index')
    plotter.figure.yaxis = ift.plotting.Axis(label='Pixel Index')

    plotter.plot.zmin = 0.
    plotter.plot.zmax = 3.
    sm = ift.SmoothingOperator.make(plot_space, sigma=0.03)
    plotter(ift.log(ift.sqrt(sm(ift.Field(plot_space, val=variance.val.real)))), path='uncertainty.html')

    plotter.plot.zmin = np.real(mock_signal.min());
    plotter.plot.zmax = np.real(mock_signal.max());
    plotter(ift.Field(plot_space, val=mock_signal.val.real), path='mock_signal.html')
    plotter(ift.Field(plot_space, val=data.val.get_full_data().real), path = 'data.html')
    plotter(ift.Field(plot_space, val=m.val.real), path = 'map.html')

