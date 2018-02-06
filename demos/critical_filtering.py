import numpy as np
import nifty4 as ift

np.random.seed(42)
# np.seterr(all="raise",under="ignore")


if __name__ == "__main__":
    # Set up position space
    #s_space = ift.RGSpace([128, 128])
    s_space = ift.HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    h_space = s_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(h_space, s_space)

    # Set up power space
    p_space = ift.PowerSpace(h_space,
                             binbounds=ift.PowerSpace.useful_binbounds(
                                 h_space, logarithmic=True))

    # Choose the prior correlation structure and defining correlation operator
    p_spec = (lambda k: (.5 / (k + 1) ** 3))
    S = ift.create_power_operator(h_space, power_spectrum=p_spec)

    # Draw a sample sh from the prior distribution in harmonic space
    sp = ift.PS_field(p_space, p_spec)
    sh = ift.power_synthesize(sp, real_signal=True)

    # Choose the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.01)
    Instrument = ift.DiagonalOperator(ift.Field(s_space, 1.))
    # Instrument._diagonal.val[200:400, 200:400] = 0
    # Instrument._diagonal.val[64:512-64, 64:512-64] = 0

    # Add a harmonic transformation to the instrument
    R = Instrument*HT

    noise = 1.
    N = ift.ScalingOperator(noise, s_space)
    n = ift.Field.from_random(domain=s_space, random_type='normal',
                              std=np.sqrt(noise), mean=0)

    # Create mock data
    d = R(sh) + n

    # The information source
    j = R.adjoint_times(N.inverse_times(d))
    realized_power = ift.log(ift.power_analyze(sh,
                                               binbounds=p_space.binbounds))
    data_power = ift.log(ift.power_analyze(HT.adjoint_times(d),
                                           binbounds=p_space.binbounds))
    d_data = d.val
    ift.plot(d, name="data.png")

    IC1 = ift.GradientNormController(name="IC1", iteration_limit=100,
                                     tol_abs_gradnorm=0.1)
    minimizer = ift.RelaxedNewton(IC1)

    ICI = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=0.1)
    map_inverter = ift.ConjugateGradient(controller=ICI)

    ICI2 = ift.GradientNormController(iteration_limit=200,
                                      tol_abs_gradnorm=1e-5)
    power_inverter = ift.ConjugateGradient(controller=ICI2)

    # Set starting position
    flat_power = ift.Field.full(p_space, 1e-8)
    m0 = ift.power_synthesize(flat_power, real_signal=True)
    t0 = ift.Field(p_space, val=-7.)

    for i in range(500):
        S0 = ift.create_power_operator(h_space, power_spectrum=ift.exp(t0))

        # Initialize non-linear Wiener Filter energy
        map_energy = ift.library.WienerFilterEnergy(
            position=m0, d=d, R=R, N=N, S=S0, inverter=map_inverter)
        # Solve the Wiener Filter analytically
        D0 = map_energy.curvature
        m0 = D0.inverse_times(j)
        # Initialize power energy with updated parameters
        power_energy = ift.library.CriticalPowerEnergy(
            position=t0, m=m0, D=D0, smoothness_prior=10., samples=3,
            inverter=power_inverter)

        power_energy = minimizer(power_energy)[0]

        # Set new power spectrum
        t0 = power_energy.position

        # Plot current estimate
        ift.dobj.mprint(i)
        if i % 50 == 0:
            ift.plot(HT(m0), name='map.png')
