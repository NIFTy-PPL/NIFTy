import nifty4 as ift
import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
    # Setting up parameters
    correlation_length = 1.     # Typical distance over which the field is correlated
    field_variance = 2.         # Variance of field in position space
    response_sigma = 0.02       # Smoothing length of response (in same unit as L)
    signal_to_noise = 100       # The signal to noise ratio
    np.random.seed(43)          # Fixing the random seed

    def power_spectrum(k):      # Defining the power spectrum
        a = 4 * correlation_length * field_variance**2
        return a / (1 + k * correlation_length) ** 4

    # Setting up the geometry
    L = 2.  # Total side-length of the domain
    N_pixels = 128  # Grid resolution (pixels per axis)
    # signal_space = ift.RGSpace([N_pixels, N_pixels], distances=L/N_pixels)
    signal_space = ift.HPSpace(16)
    harmonic_space = signal_space.get_default_codomain()
    fft = ift.FFTOperator(harmonic_space, target=signal_space)
    power_space = ift.PowerSpace(harmonic_space)

    # Creating the mock signal
    S = ift.create_power_operator(harmonic_space,
                                  power_spectrum=power_spectrum)
    mock_power = ift.PS_field(power_space, power_spectrum)
    mock_signal = fft(ift.power_synthesize(mock_power, real_signal=True))

    # Setting up an exemplary response
    mask = ift.Field.ones(signal_space)
    N10 = int(N_pixels/10)
    # mask.val[N10*5:N10*9, N10*5:N10*9] = 0.
    R = ift.ResponseOperator(signal_space, sigma=(response_sigma,),
                             exposure=(mask,))
    data_domain = R.target[0]
    R_harmonic = R*fft

    # Setting up the noise covariance and drawing a random noise realization
    ndiag = ift.Field.full(data_domain, mock_signal.var()/signal_to_noise)
    N = ift.DiagonalOperator(ndiag.weight(1))
    noise = ift.Field.from_random(domain=data_domain, random_type='normal',
        std=mock_signal.std()/np.sqrt(signal_to_noise), mean=0)
    data = R(ift.exp(mock_signal)) + noise

    # Wiener filter
    m0 = ift.Field.zeros(harmonic_space)
    ctrl = ift.GradientNormController(verbose=False, tol_abs_gradnorm=0.0001)
    ctrl2 = ift.GradientNormController(verbose=True, tol_abs_gradnorm=0.1,
                                       name="outer")
    inverter = ift.ConjugateGradient(controller=ctrl)
    energy = ift.library.LogNormalWienerFilterEnergy(m0, data, R_harmonic,
                                                     N, S, inverter=inverter)
    # minimizer1 = ift.VL_BFGS(controller=ctrl2, max_history_length=20)
    minimizer2 = ift.RelaxedNewton(controller=ctrl2)
    # minimizer3 = ift.SteepestDescent(controller=ctrl2)

    # me1 = minimizer1(energy)
    me2 = minimizer2(energy)
    # me3 = minimizer3(energy)

    # m1 = fft(me1[0].position)
    m2 = fft(me2[0].position)
    # m3 = fft(me3[0].position)

    # Plotting
    plotdict = {"xlabel": "Pixel index", "ylabel": "Pixel index",
                "colormap": "Planck-like"}
    ift.plotting.plot(mock_signal, name="mock_signal.png", **plotdict)
    logdata = np.log(ift.dobj.to_global_data(data.val)).reshape(signal_space.shape)
    ift.plotting.plot(ift.Field(signal_space,
                                val=ift.dobj.from_global_data(logdata)),
                      name="log_of_data.png", **plotdict)
    # ift.plotting.plot(m1,name='m_LBFGS.png', **plotdict)
    ift.plotting.plot(m2, name='m_Newton.png', **plotdict)
    # ift.plotting.plot(m3, name='m_SteepestDescent.png', **plotdict)

    # Probing the variance
    class Proby(ift.DiagonalProberMixin, ift.Prober):
        pass
    proby = Proby(signal_space, probe_count=1)
    proby(lambda z: fft(me2[0].curvature.inverse_times(fft.adjoint_times(z))))

    sm = ift.FFTSmoothingOperator(signal_space, sigma=0.02)
    variance = sm(proby.diagonal.weight(-1))
    ift.plotting.plot(variance, name='variance.png', **plotdict)
