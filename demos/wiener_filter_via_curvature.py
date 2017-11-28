use_nifty2go = True

import numpy as np
if use_nifty2go:
    import nifty2go as ift
else:
    import nifty as ift
import numericalunits as nu

if __name__ == "__main__":
    # In MPI mode, the random seed for numericalunits must be set by hand
    if not use_nifty2go:
        ift.nifty_configuration['default_distribution_strategy'] = 'fftw'
        ift.nifty_configuration['harmonic_rg_base'] = 'real'
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
    N_pixels = 4096
    shape = [N_pixels]*dimensionality

    signal_space = ift.RGSpace(shape, distances=L/N_pixels)
    if use_nifty2go:
        harmonic_space = signal_space.get_default_codomain()
    else:
        harmonic_space = ift.FFTOperator.get_default_codomain(signal_space)
    fft = ift.FFTOperator(harmonic_space, target=signal_space)
    power_space = ift.PowerSpace(harmonic_space)

    # Creating the mock data
    S = ift.create_power_operator(harmonic_space,
                                  power_spectrum=power_spectrum)
    np.random.seed(43)

    if use_nifty2go:
        mock_power = ift.PS_field(power_space, power_spectrum)
        mock_harmonic = ift.power_synthesize(mock_power, real_signal=True)
    else:
        mock_power = ift.Field(power_space, val=power_spectrum)
        mock_harmonic = mock_power.power_synthesize(real_signal=True)
    mock_harmonic = mock_harmonic.real
    mock_signal = fft(mock_harmonic)

    exposure = 1.
    R = ift.ResponseOperator(signal_space, sigma=(response_sigma,),
                             exposure=(exposure,))
    data_domain = R.target[0]
    if use_nifty2go:
        R_harmonic = ift.ComposedOperator([fft, R])
    else:
        R_harmonic = ift.ComposedOperator([fft, R], default_spaces=[0, 0])

    if use_nifty2go:
        N = ift.DiagonalOperator(
            ift.Field.full(data_domain,
                           mock_signal.var()/signal_to_noise).weight(1))
    else:
        ndiag = ift.Field(data_domain, mock_signal.var()/signal_to_noise).weight(1)
        N = ift.DiagonalOperator(data_domain, ndiag)

    noise = ift.Field.from_random(
        domain=data_domain, random_type='normal',
        std=mock_signal.std()/np.sqrt(signal_to_noise), mean=0)
    data = R(mock_signal) + noise

    # Wiener filter

    j = R_harmonic.adjoint_times(N.inverse_times(data))
    if use_nifty2go:
        ctrl = ift.GradientNormController(
            verbose=True, iteration_limit=10, tol_abs_gradnorm=1e-4/nu.K/(nu.m**(0.5*dimensionality)))
    else:
        def print_stats(a_energy, iteration):  # returns current energy
            x = a_energy.value
            print(x, iteration)
        ctrl = ift.GradientNormController(
            callback=print_stats, iteration_limit=10, tol_abs_gradnorm=1e-4/nu.K/(nu.m**(0.5*dimensionality)))

    inverter = ift.ConjugateGradient(controller=ctrl)
    if use_nifty2go:
        wiener_curvature = ift.library.WienerFilterCurvature(S=S, N=N,
                                                             R=R_harmonic)
        wiener_curvature = ift.InversionEnabler(wiener_curvature, inverter)
    else:
        wiener_curvature = ift.library.WienerFilterCurvature(S=S, N=N, R=R_harmonic, inverter=inverter)

    m = wiener_curvature.inverse_times(j)
    m_s = fft(m)

    sspace2 = ift.RGSpace(shape, distances=L/N_pixels/nu.m)

    ift.plotting.plot(ift.Field(sspace2, mock_signal.real.val)/nu.K,
                      name="mock_signal.pdf")
    data = ift.dobj.to_global_data(data.val.real).reshape(sspace2.shape)/nu.K
    data = ift.Field(sspace2, val=ift.dobj.from_global_data(data))/nu.K
    ift.plotting.plot(ift.Field(sspace2, val=data), name="data.pdf")
    ift.plotting.plot(ift.Field(sspace2, m_s.real.val)/nu.K, name="map.pdf")
