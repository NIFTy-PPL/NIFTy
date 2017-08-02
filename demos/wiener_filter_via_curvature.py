import numpy as np

from nifty import RGSpace, PowerSpace, Field, FFTOperator, ComposedOperator,\
                  DiagonalOperator, ResponseOperator, plotting,\
                  create_power_operator
from nifty.library import WienerFilterCurvature


if __name__ == "__main__":

    distribution_strategy = 'not'

    # Setting up variable parameters

    # Typical distance over which the field is correlated
    correlation_length = 0.01
    # Variance of field in position space sqrt(<|s_x|^2>)
    field_variance = 2.
    # smoothing length of response (in same unit as L)
    response_sigma = 0.1
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
    N_pixels = 512

    signal_space = RGSpace([N_pixels, N_pixels], distances=L/N_pixels)
    harmonic_space = FFTOperator.get_default_codomain(signal_space)
    fft = FFTOperator(harmonic_space, target=signal_space,
                      domain_dtype=np.complex, target_dtype=np.float)
    power_space = PowerSpace(harmonic_space,
                             distribution_strategy=distribution_strategy)

    # Creating the mock data
    S = create_power_operator(harmonic_space, power_spectrum=power_spectrum,
                              distribution_strategy=distribution_strategy)

    mock_power = Field(power_space, val=power_spectrum,
                       distribution_strategy=distribution_strategy)
    np.random.seed(43)
    mock_harmonic = mock_power.power_synthesize(real_signal=True)
    mock_signal = fft(mock_harmonic)

    R = ResponseOperator(signal_space, sigma=(response_sigma,))
    data_domain = R.target[0]
    R_harmonic = ComposedOperator([fft, R], default_spaces=[0, 0])

    N = DiagonalOperator(data_domain,
                         diagonal=mock_signal.var()/signal_to_noise,
                         bare=True)
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
    plotter.title = 'mock_signal.html';
    plotter(mock_signal)
    plotter.title = 'data.html'
    plotter(Field(signal_space,
                  val=data.val.get_full_data().reshape(signal_space.shape)))
    plotter.title = 'map.html'; plotter(m_s)