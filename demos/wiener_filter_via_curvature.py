import numpy as np
import nifty2go as ift
import numericalunits as nu

if __name__ == "__main__":
    # In MPI mode, the random seed for numericalunits must be set by hand
    nu.reset_units(43)
    dimensionality = 2
    np.random.seed(43)

    # Setting up variable parameters

    # Typical distance over which the field is correlated
    correlation_length = 0.05*nu.m
    # sigma of field in position space sqrt(<|s_x|^2>)
    field_sigma = 2. * nu.K
    # smoothing length of response
    response_sigma = 0.01*nu.m
    # The signal to noise ratio
    signal_to_noise = 0.7

    # note that field_variance**2 = a*k_0/4. for this analytic form of power
    # spectrum
    def power_spectrum(k):
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
    fft = ift.FFTOperator(harmonic_space, target=signal_space)
    power_space = ift.PowerSpace(harmonic_space)

    # Creating the mock data
    S = ift.create_power_operator(harmonic_space,
                                  power_spectrum=power_spectrum)
    np.random.seed(43)

    mock_power = ift.PS_field(power_space, power_spectrum)
    mock_harmonic = ift.power_synthesize(mock_power, real_signal=True)
    mock_harmonic = mock_harmonic.real
    mock_signal = fft(mock_harmonic)

    exposure = 1.
    R = ift.ResponseOperator(signal_space, sigma=(response_sigma,),
                             exposure=(exposure,))
    data_domain = R.target[0]
    R_harmonic = ift.ComposedOperator([fft, R])

    N = ift.DiagonalOperator(
        ift.Field.full(data_domain,
                       mock_signal.var()/signal_to_noise).weight(1))
    noise = ift.Field.from_random(
        domain=data_domain, random_type='normal',
        std=mock_signal.std()/np.sqrt(signal_to_noise), mean=0)
    data = R(mock_signal) + noise

    # Wiener filter

    j = R_harmonic.adjoint_times(N.inverse_times(data))
    ctrl = ift.GradientNormController(
        verbose=True, tol_abs_gradnorm=1e-4/nu.K/(nu.m**(0.5*dimensionality)))
    wiener_curvature = ift.library.WienerFilterCurvature(S=S, N=N,
                                                         R=R_harmonic)
    inverter = ift.ConjugateGradient(controller=ctrl)
    wiener_curvature = ift.InversionEnabler(wiener_curvature, inverter)

    m = wiener_curvature.inverse_times(j)
    m_s = fft(m)

    sspace2 = ift.RGSpace(shape, distances=L/N_pixels/nu.m)

    ift.plotting.plot(ift.Field(sspace2, mock_signal.real.val)/nu.K,
                      name="mock_signal.png")
    data = ift.dobj.to_global_data(data.val.real).reshape(sspace2.shape)/nu.K
    data = ift.Field(sspace2, val=ift.dobj.from_global_data(data))/nu.K
    ift.plotting.plot(ift.Field(sspace2, val=data), name="data.png")
    ift.plotting.plot(ift.Field(sspace2, m_s.real.val)/nu.K, name="map.png")
