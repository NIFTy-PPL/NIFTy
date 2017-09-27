import numpy as np

from nifty import RGSpace, PowerSpace, Field, FFTOperator, ComposedOperator,\
                  DiagonalOperator, ResponseOperator, plotting,\
                  create_power_operator, nifty_configuration
from nifty.library import WienerFilterCurvature


if __name__ == "__main__":

    nifty_configuration['default_distribution_strategy'] = 'fftw'
    nifty_configuration['harmonic_rg_base'] = 'real'

    # Setting up variable parameters

    # Typical distance over which the field is correlated
    correlation_length = 0.05
    # Variance of field in position space sqrt(<|s_x|^2>)
    field_variance = 2.
    # smoothing length of response (in same unit as L)
    response_sigma = 0.01
    # The signal to noise ratio
    signal_to_noise = 0.7

    # note that field_variance**2 = a*k_0/4. for this analytic form of power
    # spectrum
    def power_spectrum(k):
        a = 4 * correlation_length * field_variance**2
        return a / (1 + k * correlation_length) ** 4

    # Setting up the geometry

    # Total side-length of the domain
    L = 2.
    # Grid resolution (pixels per axis)
    N_pixels = 128

    signal_space = RGSpace([N_pixels, N_pixels], distances=L/N_pixels)
    harmonic_space = FFTOperator.get_default_codomain(signal_space)
    fft = FFTOperator(harmonic_space, target=signal_space)
    power_space = PowerSpace(harmonic_space)

    # Creating the mock data
    S = create_power_operator(harmonic_space, power_spectrum=power_spectrum)

    mock_power = Field(power_space, val=power_spectrum)
    np.random.seed(43)
    mock_harmonic = mock_power.power_synthesize(real_signal=True)
    if nifty_configuration['harmonic_rg_base'] == 'real':
        mock_harmonic = mock_harmonic.real
    mock_signal = fft(mock_harmonic)

    R = ResponseOperator(signal_space, sigma=(response_sigma,))
    data_domain = R.target[0]
    R_harmonic = ComposedOperator([fft, R], default_spaces=[0, 0])

    ndiag = Field(data_domain,mock_signal.var()/signal_to_noise).weight(1)
    N = DiagonalOperator(data_domain,ndiag)
    noise = Field.from_random(domain=data_domain,
                              random_type='normal',
                              std=mock_signal.std()/np.sqrt(signal_to_noise),
                              mean=0)
    data = R(mock_signal) + noise

    # Wiener filter

    j = R_harmonic.adjoint_times(N.inverse_times(data))
    wiener_curvature = WienerFilterCurvature(S=S, N=N, R=R_harmonic)

    m = wiener_curvature.inverse_times(j)
    m_s = fft(m)

    plotter = plotting.RG2DPlotter()
    plotter.path = 'mock_signal.html'
    plotter(mock_signal.real)
    plotter.path = 'data.html'
    plotter(Field(
                signal_space,
                val=data.val.get_full_data().real.reshape(signal_space.shape)))
    plotter.path = 'map.html'
    plotter(m_s.real)
