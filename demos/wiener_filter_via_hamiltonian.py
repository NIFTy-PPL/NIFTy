import nifty4 as ift
import numpy as np

np.random.seed(42)


if __name__ == "__main__":
    # Set up position space
    s_space = ift.RGSpace([128, 128])
    # s_space = ift.HPSpace(32)

    # Define associated harmonic space and harmonic transformation
    h_space = s_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(h_space, s_space)

    # Setting up power space
    p_space = ift.PowerSpace(h_space)

    # Choosing the prior correlation structure and defining
    # correlation operator
    p_spec = (lambda k: (42/(k+1)**3))

    S = ift.create_power_operator(h_space, power_spectrum=p_spec)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = ift.PS_field(p_space, p_spec)
    sh = ift.power_synthesize(sp, real_signal=True)

    # Choosing the measurement instrument
    diag = np.ones(s_space.shape)
    diag[20:80, 20:80] = 0
    diag = ift.Field(s_space, ift.dobj.from_global_data(diag))
    Instrument = ift.DiagonalOperator(diag)

    # Adding a harmonic transformation to the instrument
    R = Instrument*ht
    noiseless_data = R(sh)
    signal_to_noise = 1.
    noise_amplitude = noiseless_data.val.std()/signal_to_noise
    N = ift.DiagonalOperator(ift.Field.full(s_space, noise_amplitude**2))
    n = ift.Field.from_random(domain=s_space,
                              random_type='normal',
                              std=noise_amplitude,
                              mean=0)

    # Creating the mock data
    d = noiseless_data + n
    j = R.adjoint_times(N.inverse_times(d))

    # Choosing the minimization strategy

    ctrl = ift.GradientNormController(name="inverter", tol_abs_gradnorm=0.1)
    inverter = ift.ConjugateGradient(controller=ctrl)
    controller = ift.GradientNormController(name="min", tol_abs_gradnorm=0.1)
    minimizer = ift.RelaxedNewton(controller=controller)
    m0 = ift.Field.zeros(h_space)
    # Initializing the Wiener Filter energy
    energy = ift.library.WienerFilterEnergy(position=m0, d=d, R=R, N=N, S=S,
                                            inverter=inverter)

    energy, convergence = minimizer(energy)
    m = energy.position
    D = energy.curvature
    ift.plot(ht(sh), name="signal.png", colormap="Planck-like")
    ift.plot(ht(m), name="m.png", colormap="Planck-like")

    # sampling the uncertainty map
    sample_variance = ift.Field.zeros(s_space)
    sample_mean = ift.Field.zeros(s_space)

    mean, variance = ift.probe_with_posterior_samples(D, ht, 50)
    ift.plot(variance, name="variance.png", colormap="Planck-like")
    ift.plot(mean, name="mean.png", colormap="Planck-like")
