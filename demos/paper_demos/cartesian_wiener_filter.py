import numpy as np
import nifty2go as ift

if __name__ == "__main__":
    signal_to_noise = 1.5  # The signal to noise ratio

    # Setting up parameters
    correlation_length_1 = 1.  # Typical distance over which the field is correlated
    field_variance_1 = 2.  # Variance of field in position space

    response_sigma_1 = 0.05  # Smoothing length of response (in same unit as L)

    def power_spectrum_1(k):  # note: field_variance**2 = a*k_0/4.
        a = 4 * correlation_length_1 * field_variance_1**2
        return a / (1 + k * correlation_length_1) ** 4.

    # Setting up the geometry
    L_1 = 2.  # Total side-length of the domain
    N_pixels_1 = 512  # Grid resolution (pixels per axis)

    signal_space_1 = ift.RGSpace([N_pixels_1], distances=L_1/N_pixels_1)
    harmonic_space_1 = signal_space_1.get_default_codomain()
    # Setting up the geometry
    L_2 = 2.  # Total side-length of the domain
    N_pixels_2 = 512  # Grid resolution (pixels per axis)
    signal_space_2 = ift.RGSpace([N_pixels_2], distances=L_2/N_pixels_2)
    harmonic_space_2 = signal_space_2.get_default_codomain()

    signal_domain = ift.DomainTuple.make((signal_space_1, signal_space_2))
    mid_domain = ift.DomainTuple.make((signal_space_1, harmonic_space_2))
    harmonic_domain = ift.DomainTuple.make((harmonic_space_1,
                                            harmonic_space_2))

    fft_1 = ift.FFTOperator(harmonic_domain, space=0)
    power_space_1 = ift.PowerSpace(harmonic_space_1)

    mock_power_1 = ift.PS_field(power_space_1, power_spectrum_1)

    # Setting up parameters
    correlation_length_2 = 1.  # Typical distance over which the field is correlated
    field_variance_2 = 2.  # Variance of field in position space

    response_sigma_2 = 0.01  # Smoothing length of response (in same unit as L)

    def power_spectrum_2(k):  # note: field_variance**2 = a*k_0/4.
        a = 4 * correlation_length_2 * field_variance_2**2
        return a / (1 + k * correlation_length_2) ** 2.5

    fft_2 = ift.FFTOperator(mid_domain, space=1)
    power_space_2 = ift.PowerSpace(harmonic_space_2)

    mock_power_2 = ift.PS_field(power_space_2, power_spectrum_2)

    fft = ift.ComposedOperator((fft_1, fft_2))

    mock_power = ift.Field(domain=(power_space_1, power_space_2),
                           val=ift.dobj.from_global_data(np.outer(ift.dobj.to_global_data(mock_power_1.val), ift.dobj.to_global_data(mock_power_2.val))))

    diagonal = ift.power_synthesize_special(mock_power, spaces=(0, 1))**2
    diagonal = diagonal.real

    S = ift.DiagonalOperator(diagonal)

    np.random.seed(10)
    mock_signal = fft(ift.power_synthesize(mock_power, real_signal=True))

    # Setting up a exemplary response
    N1_10 = int(N_pixels_1/10)
    mask_1 = np.ones(signal_space_1.shape)
    mask_1[N1_10*7:N1_10*9] = 0.
    mask_1 = ift.Field(signal_space_1, ift.dobj.from_global_data(mask_1))

    N2_10 = int(N_pixels_2/10)
    mask_2 = np.ones(signal_space_2.shape)
    mask_2[N2_10*7:N2_10*9] = 0.
    mask_2 = ift.Field(signal_space_2, ift.dobj.from_global_data(mask_2))

    R = ift.ResponseOperator(signal_domain, spaces=(0, 1),
                             sigma=(response_sigma_1, response_sigma_2),
                             exposure=(mask_1, mask_2))
    data_domain = R.target
    R_harmonic = ift.ComposedOperator([fft, R])

    # Setting up the noise covariance and drawing a random noise realization
    ndiag = ift.Field.full(data_domain, mock_signal.var()/signal_to_noise)
    N = ift.DiagonalOperator(ndiag.weight(1))
    noise = ift.Field.from_random(domain=data_domain, random_type='normal',
                                  std=mock_signal.std()/np.sqrt(signal_to_noise),
                                  mean=0)
    data = R(mock_signal) + noise

    # Wiener filter
    j = R_harmonic.adjoint_times(N.inverse_times(data))
    ctrl = ift.GradientNormController(verbose=True, tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=ctrl)
    wiener_curvature = ift.library.WienerFilterCurvature(S=S, N=N,
                                                         R=R_harmonic)
    wiener_curvature = ift.InversionEnabler(wiener_curvature, inverter)

    m_k = wiener_curvature.inverse_times(j)
    m = fft(m_k)

    # Probing the variance
    class Proby(ift.DiagonalProberMixin, ift.Prober):
        pass
    proby = Proby((signal_space_1, signal_space_2), probe_count=1, ncpu=1)
    proby(lambda z: fft(wiener_curvature.inverse_times(fft.inverse_times(z))))
#    sm = SmoothingOperator(signal_space, sigma=0.02)
#    variance = sm(proby.diagonal.weight(-1))
    variance = proby.diagonal.weight(-1)

    plot_space = ift.RGSpace((N_pixels_1, N_pixels_2))
    sm = ift.FFTSmoothingOperator(plot_space, sigma=0.03)
    ift.plotting.plot(ift.log(ift.sqrt(sm(ift.Field(plot_space, val=variance.val.real)))), name='uncertainty.png',zmin=0.,zmax=3.,title="Uncertainty map",colormap="Planck-like")
    ift.plotting.plot(ift.Field(plot_space, val=mock_signal.val.real), name='mock_signal.png',colormap="Planck-like")
    ift.plotting.plot(ift.Field(plot_space, val=data.val.real), name='data.png',colormap="Planck-like")
    ift.plotting.plot(ift.Field(plot_space, val=m.val.real), name='map.png',colormap="Planck-like")
