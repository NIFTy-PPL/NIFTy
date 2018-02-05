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
    power_space = ift.PowerSpace(
        harmonic_space, binbounds=ift.PowerSpace.useful_binbounds(
            harmonic_space, logarithmic=True))

    # Creating the mock signal
    S = ift.create_power_operator(harmonic_space,
                                  power_spectrum=power_spectrum)
    mock_power = ift.PS_field(power_space, power_spectrum)
    mock_signal = ift.power_synthesize(mock_power, real_signal=True)

    # Setting up an exemplary response
    mask = np.ones(signal_space.shape)
    N10 = int(N_pixels/10)
    mask[N10*5:N10*9, N10*5:N10*9] = 0.
    mask = ift.Field(signal_space, ift.dobj.from_global_data(mask))
    R = ift.GeometryRemover(signal_space)
    R = R*ift.DiagonalOperator(mask)
    R = R*ht
    R = R * ift.create_harmonic_smoothing_operator((harmonic_space,),0,response_sigma)
    data_domain = R.target[0]

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
    ctrl = ift.GradientNormController(name="inverter", tol_abs_gradnorm=1e-2)
    inverter = ift.ConjugateGradient(controller=ctrl)
    wiener_curvature = ift.library.WienerFilterCurvature(
        S=S, N=N, R=R, inverter=inverter)
    m_k = wiener_curvature.inverse_times(j)
    m = ht(m_k)

    plotdict = {"xlabel": "Pixel index", "ylabel": "Pixel index",
                "colormap": "Planck-like"}
    ift.plot(ht(mock_signal), name="mock_signal.png", **plotdict)
    ift.plot(ift.Field(signal_space, val=data.val),
             name="data.png", **plotdict)
    ift.plot(m, name="map.png", **plotdict)

    # sampling the uncertainty map
    sample_variance = ift.Field.zeros(signal_space)
    sample_mean = ift.Field.zeros(signal_space)
    n_samples = 10
    for i in range(n_samples):
        sample = ht(wiener_curvature.draw_sample()) + m
        sample_variance += sample**2
        sample_mean += sample
    variance = sample_variance/n_samples - (sample_mean/n_samples)**2
    ift.plot(ift.sqrt(variance), name="uncertainty.png", **plotdict)
