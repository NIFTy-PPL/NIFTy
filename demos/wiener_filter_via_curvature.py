import numpy as np
import nifty4 as ift
import numericalunits as nu


if __name__ == "__main__":
    # In MPI mode, the random seed for numericalunits must be set by hand
    #nu.reset_units(43)
    dimensionality = 2
    np.random.seed(43)

    # Setting up variable parameters

    # Typical distance over which the field is correlated
    correlation_length = 0.05*nu.m
    # sigma of field in position space sqrt(<|s_x|^2>)
    field_sigma = 2. * nu.K
    # smoothing length of response
    response_sigma = 0.01*nu.m
    # The signal to noise ratio ***CURRENTLY BROKEN***
    signal_to_noise = 70.7

    # note that field_variance**2 = a*k_0/4. for this analytic form of power
    # spectrum
    def power_spectrum(k):
    	#RL FIXME: signal_amplitude is not how much signal varies
        cldim = correlation_length**(2*dimensionality)
        a = 4/(2*np.pi) * cldim * field_sigma**2
        # to be integrated over spherical shells later on
        return a / (1 + (k*correlation_length)**(2*dimensionality)) ** 2

    # Setting up the geometry

    # Total side-length of the domain
    L = 2.*nu.m
    # Grid resolution (pixels per axis)
    N_pixels = 512
    shape = [N_pixels]*dimensionality

    signal_space = ift.RGSpace(shape, distances=L/N_pixels)
    harmonic_space = signal_space.get_default_codomain()
    fft = ift.HarmonicTransformOperator(harmonic_space, target=signal_space)
    power_space = ift.PowerSpace(harmonic_space)

    # Creating the mock data
    S = ift.create_power_operator(harmonic_space,
                                  power_spectrum=power_spectrum)
    np.random.seed(43)

    mock_power = ift.PS_field(power_space, power_spectrum)
    mock_signal = ift.power_synthesize(mock_power, real_signal=True)

    sensitivity = (1./nu.m)**dimensionality/nu.K
    R = ift.GeometryRemover(signal_space)
    R = R*ift.ScalingOperator(sensitivity, signal_space)
    R = R*fft
    R = R * ift.create_harmonic_smoothing_operator((harmonic_space,),0,response_sigma)
    data_domain = R.target[0]

    noise_amplitude = 1./signal_to_noise*field_sigma*sensitivity*((L/N_pixels)**dimensionality)
    print "noise amplitude:", noise_amplitude
    N = ift.DiagonalOperator(
        ift.Field.full(data_domain, noise_amplitude**2))
    noise = ift.Field.from_random(
        domain=data_domain, random_type='normal',
        std=noise_amplitude, mean=0)
    data = R(mock_signal) + noise
     # Wiener filter

    j = R.adjoint_times(N.inverse_times(data))
    ctrl = ift.GradientNormController(
        name="inverter", tol_abs_gradnorm=1e-5/(nu.K*(nu.m**dimensionality)))
    inverter = ift.ConjugateGradient(controller=ctrl)
    wiener_curvature = ift.library.WienerFilterCurvature(
        S=S, N=N, R=R, inverter=inverter)

    m = wiener_curvature.inverse_times(j)
    m_s = fft(m)

    sspace2 = ift.RGSpace(shape, distances=L/N_pixels/nu.m)

    ift.plot(ift.Field(sspace2, fft(mock_signal).val)/nu.K, name="mock_signal.png")
    #data = ift.dobj.to_global_data(data.val).reshape(sspace2.shape)
    #data = ift.Field(sspace2, val=ift.dobj.from_global_data(data))
    ift.plot(ift.Field(sspace2, val=fft(R.adjoint_times(data)).val), name="data.png")
    print "msig",np.min(fft(mock_signal).val)/nu.K, np.max(fft(mock_signal).val)/nu.K
    print "map",np.min(m_s.val)/nu.K, np.max(m_s.val)/nu.K
    ift.plot(ift.Field(sspace2, m_s.val)/nu.K, name="map.png")
