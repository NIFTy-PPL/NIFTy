import nifty4 as ift
import numpy as np


if __name__ == "__main__":
    # Setting up parameters
    L = 2.                         # Total side-length of the domain
    N_pixels = 512                 # Grid resolution (pixels per axis)
    correlation_length_scale = .2  # Typical distance over which the field is
                                   # correlated
    fluctuation_scale = 2.         # Variance of field in position space
    response_sigma = 0.05          # Smoothing length of response
    signal_to_noise = 1.5          # The signal to noise ratio
    np.random.seed(43)             # Fixing the random seed

    def power_spectrum(k):         # Defining the power spectrum
        a = 4 * correlation_length_scale * fluctuation_scale**2
        return a / (1 + (k * correlation_length_scale)**2) ** 2

    signal_space = ift.RGSpace([N_pixels, N_pixels], distances=L/N_pixels)
    harmonic_space = signal_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(harmonic_space, target=signal_space)

    # Creating the mock signal
    S = ift.create_power_operator(harmonic_space,
                                  power_spectrum=power_spectrum)
    mock_signal = S.draw_sample()

    # Setting up an exemplary response
    mask = np.ones(signal_space.shape)
    N10 = int(N_pixels/10)
    mask[N10*5:N10*9, N10*5:N10*9] = 0.
    mask = ift.Field.from_global_data(signal_space, mask).lock()
    R = ift.GeometryRemover(signal_space)
    R = R*ift.DiagonalOperator(mask)
    R = R*ht
    R = R * ift.create_harmonic_smoothing_operator((harmonic_space,), 0,
                                                   response_sigma)
    data_domain = R.target[0]

    noiseless_data = R(mock_signal)
    noise_amplitude = noiseless_data.val.std()/signal_to_noise
    # Setting up the noise covariance and drawing a random noise realization
    N = ift.ScalingOperator(noise_amplitude**2, data_domain)
    noise = N.draw_sample()
    data = noiseless_data + noise

    # Wiener filter
    j = R.adjoint_times(N.inverse_times(data))
    ctrl = ift.GradientNormController(name="inverter", tol_abs_gradnorm=1e-2)
    sampling_ctrl = ift.GradientNormController(name="sampling",
                                               tol_abs_gradnorm=2e1)
    inverter = ift.ConjugateGradient(controller=ctrl)
    sampling_inverter = ift.ConjugateGradient(controller=sampling_ctrl)
    wiener_curvature = ift.library.WienerFilterCurvature(
        S=S, N=N, R=R, inverter=inverter, sampling_inverter=sampling_inverter)
    m_k = wiener_curvature.inverse_times(j)
    m = ht(m_k)

    plotdict = {"colormap": "Planck-like"}
    ift.plot(ht(mock_signal), name="mock_signal.png", **plotdict)
    ift.plot(data.cast_domain(signal_space), name="data.png", **plotdict)
    ift.plot(m, name="map.png", **plotdict)

    # sampling the uncertainty map
    mean, variance = ift.probe_with_posterior_samples(wiener_curvature, ht, 50)
    ift.plot(ift.sqrt(variance), name="uncertainty.png", **plotdict)
    ift.plot(mean+m, name="posterior_mean.png", **plotdict)
