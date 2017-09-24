import numpy as np
import nifty2go as ift


if __name__ == "__main__":
    np.random.seed(43)

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
    N_pixels = 512

    signal_space = ift.RGSpace([N_pixels, N_pixels], distances=L/N_pixels)
    harmonic_space = signal_space.get_default_codomain()
    fft = ift.FFTOperator(harmonic_space, target=signal_space)
    power_space = ift.PowerSpace(harmonic_space)

    # Creating the mock data
    S = ift.create_power_operator(harmonic_space, power_spectrum=power_spectrum)
    np.random.seed(43)

    mock_power = ift.Field(power_space, val=power_spectrum(power_space.k_lengths))
    mock_harmonic = mock_power.power_synthesize(real_signal=True)
    mock_harmonic = mock_harmonic.real
    mock_signal = fft(mock_harmonic)

    R = ift.ResponseOperator(signal_space, sigma=(response_sigma,))
    data_domain = R.target[0]
    R_harmonic = ift.ComposedOperator([fft, R], default_spaces=[0, 0])

    N = ift.DiagonalOperator(ift.Field(data_domain,mock_signal.var()/signal_to_noise).weight(1))
    noise = ift.Field.from_random(domain=data_domain,
                              random_type='normal',
                              std=mock_signal.std()/np.sqrt(signal_to_noise),
                              mean=0)
    data = R(mock_signal) + noise

    # Wiener filter

    j = R_harmonic.adjoint_times(N.inverse_times(data))
    ctrl = ift.DefaultIterationController(verbose=True,tol_abs_gradnorm=1e-2)
    inverter = ift.ConjugateGradient(controller=ctrl)
    wiener_curvature = ift.library.WienerFilterCurvature(S=S, N=N, R=R_harmonic, inverter=inverter)

    m = wiener_curvature.inverse_times(j)
    m_s = fft(m)

    ift.plotting.plot(mock_signal.real,name="mock_signal.pdf")
    ift.plotting.plot(ift.Field(signal_space,
                val=data.val.real.reshape(signal_space.shape)), name="data.pdf")
    ift.plotting.plot(m_s.real, name="map.pdf")
