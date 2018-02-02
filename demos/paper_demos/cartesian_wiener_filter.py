import numpy as np
import nifty4 as ift

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

    signal_space_1 = ift.RGSpace([N_pixels_1], distances=L_1/N_pixels_1)
    harmonic_space_1 = signal_space_1.get_default_codomain()
    signal_space_2 = ift.RGSpace([N_pixels_2], distances=L_2/N_pixels_2)
    harmonic_space_2 = signal_space_2.get_default_codomain()

    signal_domain = ift.DomainTuple.make((signal_space_1, signal_space_2))
    mid_domain = ift.DomainTuple.make((signal_space_1, harmonic_space_2))
    harmonic_domain = ift.DomainTuple.make((harmonic_space_1,
                                            harmonic_space_2))

    ht_1 = ift.HarmonicTransformOperator(harmonic_domain, space=0)
    power_space_1 = ift.PowerSpace(harmonic_space_1)

    mock_power_1 = ift.PS_field(power_space_1, power_spectrum_1)

    def power_spectrum_2(k):  # note: field_variance**2 = a*k_0/4.
        a = 4 * correlation_length_2 * field_variance_2**2
        return a / (1 + k * correlation_length_2) ** 2.5

    ht_2 = ift.HarmonicTransformOperator(mid_domain, space=1)
    power_space_2 = ift.PowerSpace(harmonic_space_2)

    mock_power_2 = ift.PS_field(power_space_2, power_spectrum_2)

    ht = ht_2*ht_1

    mock_power = ift.Field(
        (power_space_1, power_space_2),
        val=ift.dobj.from_global_data(
            np.outer(ift.dobj.to_global_data(mock_power_1.val),
                     ift.dobj.to_global_data(mock_power_2.val))))

    diagonal = ift.power_synthesize_nonrandom(mock_power, spaces=(0, 1))**2

    S = ift.DiagonalOperator(diagonal)

    np.random.seed(10)
    mock_signal = ift.power_synthesize(mock_power, real_signal=True)

    # Setting up a exemplary response
    N1_10 = int(N_pixels_1/10)
    mask_1 = np.ones(signal_space_1.shape)
    mask_1[N1_10*7:N1_10*9] = 0.
    mask_1 = ift.Field(signal_space_1, ift.dobj.from_global_data(mask_1))

    N2_10 = int(N_pixels_2/10)
    mask_2 = np.ones(signal_space_2.shape)
    mask_2[N2_10*7:N2_10*9] = 0.
    mask_2 = ift.Field(signal_space_2, ift.dobj.from_global_data(mask_2))

    R = ift.GeometryRemover(signal_domain)
    R = R*ift.DiagonalOperator(mask_1, signal_domain,spaces=0)
    R = R*ift.DiagonalOperator(mask_2, signal_domain,spaces=1)
    R = R*ht
    R = R * ift.create_harmonic_smoothing_operator(harmonic_domain, 0,
                                                   response_sigma_1)
    R = R * ift.create_harmonic_smoothing_operator(harmonic_domain, 1,
                                                   response_sigma_2)
    data_domain = R.target

    noiseless_data = R(mock_signal)
    noise_amplitude = noiseless_data.val.std()/signal_to_noise
    # Setting up the noise covariance and drawing a random noise realization
    ndiag = ift.Field.full(data_domain, noise_amplitude**2)
    N = ift.DiagonalOperator(ndiag)
    noise = ift.Field.from_random(
        domain=data_domain, random_type='normal',
        std=noise_amplitude, mean=0)
    data = noiseless_data + noise

    # Wiener filter
    j = R.adjoint_times(N.inverse_times(data))
    ctrl = ift.GradientNormController(name="inverter", tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=ctrl)
    wiener_curvature = ift.library.WienerFilterCurvature(
        S=S, N=N, R=R, inverter=inverter)

    m_k = wiener_curvature.inverse_times(j)
    m = ht(m_k)

    plotdict = {"xlabel": "Pixel index", "ylabel": "Pixel index",
                "colormap": "Planck-like"}
    plot_space = ift.RGSpace((N_pixels_1, N_pixels_2))
    ift.plot(ift.Field(plot_space,val=ht(mock_signal).val), name='mock_signal.png',
             **plotdict)
    ift.plot(ift.Field(plot_space,val=data.val), name='data.png', **plotdict)
    ift.plot(ift.Field(plot_space,val=m.val), name='map.png', **plotdict)
    # sampling the uncertainty map
    sample_variance = ift.Field.zeros(signal_domain)
    sample_mean = ift.Field.zeros(signal_domain)
    n_samples = 10
    for i in range(n_samples):
        sample = ht(wiener_curvature.generate_posterior_sample()) + m
        sample_variance += sample**2
        sample_mean += sample
    variance = sample_variance/n_samples - (sample_mean/n_samples)**2
    ift.plot(ift.Field(plot_space, val=ift.sqrt(variance).val), name="uncertainty.png", **plotdict)
