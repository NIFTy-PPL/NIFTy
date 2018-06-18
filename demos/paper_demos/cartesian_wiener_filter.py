import numpy as np
import nifty5 as ift

if __name__ == "__main__":
    signal_to_noise = 0.5  # The signal to noise ratio

    # Setting up parameters
    L_1 = 2.                   # Total side-length of the domain
    N_pixels_1 = 512           # Grid resolution (pixels per axis)
    L_2 = 2.                   # Total side-length of the domain
    N_pixels_2 = 512           # Grid resolution (pixels per axis)
    correlation_length_1 = 1.
    field_variance_1 = 2.      # Variance of field in position space
    response_sigma_1 = 0.05    # Smoothing length of response
    correlation_length_2 = 1.
    field_variance_2 = 2.      # Variance of field in position space
    response_sigma_2 = 0.01    # Smoothing length of response

    def power_spectrum_1(k):   # note: field_variance**2 = a*k_0/4.
        a = 4 * correlation_length_1 * field_variance_1**2
        return a / (1 + k * correlation_length_1) ** 4.

    def power_spectrum_2(k):  # note: field_variance**2 = a*k_0/4.
        a = 4 * correlation_length_2 * field_variance_2**2
        return a / (1 + k * correlation_length_2) ** 2.5

    signal_space_1 = ift.RGSpace([N_pixels_1], distances=L_1/N_pixels_1)
    harmonic_space_1 = signal_space_1.get_default_codomain()
    signal_space_2 = ift.RGSpace([N_pixels_2], distances=L_2/N_pixels_2)
    harmonic_space_2 = signal_space_2.get_default_codomain()

    signal_domain = ift.DomainTuple.make((signal_space_1, signal_space_2))
    harmonic_domain = ift.DomainTuple.make((harmonic_space_1,
                                            harmonic_space_2))

    ht_1 = ift.HarmonicTransformOperator(harmonic_domain, space=0)
    ht_2 = ift.HarmonicTransformOperator(ht_1.target, space=1)
    ht = ht_2*ht_1

    S = (ift.create_power_operator(harmonic_domain, power_spectrum_1, 0) *
         ift.create_power_operator(harmonic_domain, power_spectrum_2, 1))

    np.random.seed(10)
    mock_signal = S.draw_sample()

    # Setting up a exemplary response
    N1_10 = int(N_pixels_1/10)
    mask_1 = np.ones(signal_space_1.shape)
    mask_1[N1_10*7:N1_10*9] = 0.
    mask_1 = ift.Field.from_global_data(signal_space_1, mask_1)

    N2_10 = int(N_pixels_2/10)
    mask_2 = np.ones(signal_space_2.shape)
    mask_2[N2_10*7:N2_10*9] = 0.
    mask_2 = ift.Field.from_global_data(signal_space_2, mask_2)

    R = ift.GeometryRemover(signal_domain)
    R = R*ift.DiagonalOperator(mask_1, signal_domain, spaces=0)
    R = R*ift.DiagonalOperator(mask_2, signal_domain, spaces=1)
    R = R*ht
    R = R * ift.create_harmonic_smoothing_operator(harmonic_domain, 0,
                                                   response_sigma_1)
    R = R * ift.create_harmonic_smoothing_operator(harmonic_domain, 1,
                                                   response_sigma_2)
    data_domain = R.target

    noiseless_data = R(mock_signal)
    noise_amplitude = noiseless_data.val.std()/signal_to_noise
    # Setting up the noise covariance and drawing a random noise realization
    N = ift.ScalingOperator(noise_amplitude**2, data_domain)
    noise = N.draw_sample()
    data = noiseless_data + noise

    # Wiener filter
    j = R.adjoint_times(N.inverse_times(data))
    ctrl = ift.GradientNormController(name="inverter", tol_abs_gradnorm=0.1)
    sampling_ctrl = ift.GradientNormController(name="sampling",
                                               tol_abs_gradnorm=1e2)
    inverter = ift.ConjugateGradient(controller=ctrl)
    sampling_inverter = ift.ConjugateGradient(controller=sampling_ctrl)
    wiener_curvature = ift.library.WienerFilterCurvature(
        S=S, N=N, R=R, inverter=inverter, sampling_inverter=sampling_inverter)

    m_k = wiener_curvature.inverse_times(j)
    m = ht(m_k)

    plotdict = {"colormap": "Planck-like"}
    plot_space = ift.RGSpace((N_pixels_1, N_pixels_2))
    ift.plot(ht(mock_signal).cast_domain(plot_space),
             name='mock_signal.png', **plotdict)
    ift.plot(data.cast_domain(plot_space), name='data.png', **plotdict)
    ift.plot(m.cast_domain(plot_space), name='map.png', **plotdict)
    # sampling the uncertainty map
    mean, variance = ift.probe_with_posterior_samples(wiener_curvature, ht, 50)
    ift.plot(ift.sqrt(variance).cast_domain(plot_space),
             name="uncertainty.png", **plotdict)
    ift.plot((mean+m).cast_domain(plot_space),
             name="posterior_mean.png", **plotdict)
