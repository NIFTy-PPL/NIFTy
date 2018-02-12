import nifty4 as ift
import numpy as np

np.random.seed(42)


if __name__ == "__main__":
    # Set up position space
    # s_space = ift.RGSpace([128, 128])
    s_space = ift.HPSpace(32)

    # Define associated harmonic space and harmonic transformation
    h_space = s_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(h_space, s_space)

    # Set up power space
    p_space = ift.PowerSpace(h_space)

    # Choose prior correlation structure and define correlation operator
    p_spec = (lambda k: (42/(k+1)**3))

    S = ift.create_power_operator(h_space, power_spectrum=p_spec)

    # Draw sample sh from the prior distribution in harmonic space
    sp = ift.PS_field(p_space, p_spec)
    sh = ift.power_synthesize(sp, real_signal=True)

    # Choose measurement instrument
    diag = np.ones(s_space.shape)
    if len(s_space.shape) == 1:
        diag[3000:7000] = 0
    elif len(s_space.shape) == 2:
        diag[20:80, 20:80] = 0
    else:
        raise NotImplementedError

    diag = ift.Field(s_space, ift.dobj.from_global_data(diag))
    Instrument = ift.DiagonalOperator(diag)

    # Add harmonic transformation to the instrument
    R = Instrument*ht
    noiseless_data = R(sh)
    signal_to_noise = 1.
    noise_amplitude = noiseless_data.val.std()/signal_to_noise
    N = ift.ScalingOperator(noise_amplitude**2, s_space)
    n = ift.Field.from_random(domain=s_space,
                              random_type='normal',
                              std=noise_amplitude,
                              mean=0)

    # Create mock data
    d = noiseless_data + n
    j = R.adjoint_times(N.inverse_times(d))

    # Choose minimization strategy
    ctrl = ift.GradientNormController(name="inverter", tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=ctrl)
    controller = ift.GradientNormController(name="min", tol_abs_gradnorm=0.1)
    minimizer = ift.RelaxedNewton(controller=controller)
    m0 = ift.Field.zeros(h_space)

    # Initialize Wiener filter energy
    energy = ift.library.WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S,
                                            inverter=inverter)

    energy, convergence = minimizer(energy)
    m = energy.position
    curv = energy.curvature

    # Generate plots
    zmax = max(ht(sh).max(), ht(m).max())
    zmin = min(ht(sh).min(), ht(m).min())
    plotdict = {"zmax": zmax, "zmin": zmin, "colormap": "Planck-like"}
    ift.plot(ht(sh), name="mock_signal.png", **plotdict)
    ift.plot(ht(m), name="reconstruction.png", **plotdict)

    # Sample uncertainty map
    sample_variance = ift.Field.zeros(s_space)
    sample_mean = ift.Field.zeros(s_space)

    mean, variance = ift.probe_with_posterior_samples(curv, ht, 50)
    ift.plot(variance, name="posterior_variance.png", **plotdict)
    ift.plot(mean+ht(m), name="posterior_mean.png", **plotdict)

    # try to do the same with diagonal probing
    variance = ift.probe_diagonal(ht*curv.inverse*ht.adjoint, 100)
    #sm = ift.FFTSmoothingOperator(s_space, sigma=0.015)
    ift.plot(variance, name="posterior_variance2.png", **plotdict)
