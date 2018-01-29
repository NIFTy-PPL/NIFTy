import nifty4 as ift
import numpy as np


if __name__ == "__main__":
    # Setting up parameters
    L = 2.                         # Total side-length of the domain
    N_pixels = 512                 # Grid resolution (pixels per axis)
    correlation_length_scale = 1.  # Typical distance over which the field is
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
    fft = ift.FFTOperator(harmonic_space, target=signal_space)
    power_space = ift.PowerSpace(
        harmonic_space, binbounds=ift.PowerSpace.useful_binbounds(
            harmonic_space, logarithmic=True))

    # Creating the mock signal
    S = ift.create_power_operator(harmonic_space,
                                  power_spectrum=power_spectrum)
    mock_power = ift.PS_field(power_space, power_spectrum)
    mock_signal = fft(ift.power_synthesize(mock_power, real_signal=True))

    # Setting up an exemplary response
    mask = np.ones(signal_space.shape)
    N10 = int(N_pixels/10)
    mask[N10*5:N10*9, N10*5:N10*9] = 0.
    mask = ift.Field(signal_space, ift.dobj.from_global_data(mask))
    R = ift.ResponseOperator(signal_space, sigma=(response_sigma,),
                             sensitivity=(mask,))
    data_domain = R.target[0]
    R_harmonic = R * fft

    # Setting up the noise covariance and drawing a random noise realization
    ndiag = 1e-8*ift.Field.full(data_domain, mock_signal.var()/signal_to_noise)
    N = ift.DiagonalOperator(ndiag)
    noise = ift.Field.from_random(
        domain=data_domain, random_type='normal',
        std=mock_signal.std()/np.sqrt(signal_to_noise), mean=0)
    data = R(mock_signal) + noise

    # Wiener filter
    j = R_harmonic.adjoint_times(N.inverse_times(data))
    ctrl = ift.GradientNormController(verbose=True, tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=ctrl)
    wiener_curvature = ift.library.WienerFilterCurvature(
        S=S, N=N, R=R_harmonic, inverter=inverter)
    m_k = wiener_curvature.inverse_times(j)
    m = fft(m_k)

    # Probing the uncertainty
    class Proby(ift.DiagonalProberMixin, ift.Prober):
        pass
    proby = Proby(signal_space, probe_count=1, ncpu=1)
    proby(lambda z: fft(wiener_curvature.inverse_times(fft.inverse_times(z))))

    sm = ift.FFTSmoothingOperator(signal_space, sigma=0.03)
    variance = ift.sqrt(sm(proby.diagonal.weight(-1)))

    # Plotting
    plotdict = {"xlabel": "Pixel index", "ylabel": "Pixel index",
                "colormap": "Planck-like"}
    ift.plot(variance, name="uncertainty.png", **plotdict)
    ift.plot(mock_signal, name="mock_signal.png", **plotdict)
    ift.plot(ift.Field(signal_space, val=data.val),
             name="data.png", **plotdict)
    ift.plot(m, name="map.png", **plotdict)
