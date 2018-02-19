import nifty4 as ift
from nifty4.library.nonlinearities import Linear
import numpy as np
np.random.seed(42)


def adjust_zero_mode(m0, t0):
    mtmp = m0.to_global_data().copy()
    zero_position = len(m0.shape)*(0,)
    zero_mode = mtmp[zero_position]
    mtmp[zero_position] = zero_mode / abs(zero_mode)
    ttmp = t0.to_global_data().copy()
    ttmp[0] += 2 * np.log(abs(zero_mode))
    return (ift.Field.from_global_data(m0.domain, mtmp),
            ift.Field.from_global_data(t0.domain, ttmp))

if __name__ == "__main__":

    noise_level = 1.
    p_spec = (lambda k: (.5 / (k + 1) ** 3))

    nonlinearity = Linear()
    # Set up position space
    s_space = ift.RGSpace((128, 128))
    h_space = s_space.get_default_codomain()

    # Define harmonic transformation and associated harmonic space
    HT = ift.HarmonicTransformOperator(h_space, target=s_space)

    # Setting up power space
    p_space = ift.PowerSpace(h_space,
                             binbounds=ift.PowerSpace.useful_binbounds(
                                 h_space, logarithmic=True))
    s_spec = ift.Field.full(p_space, 1.)
    # Choosing the prior correlation structure and defining
    # correlation operator
    p = ift.PS_field(p_space, p_spec)
    log_p = ift.log(p)
    S = ift.create_power_operator(h_space, power_spectrum=s_spec)

    # Drawing a sample sh from the prior distribution in harmonic space
    sh = ift.power_synthesize(s_spec)

    # Choosing the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.01)
    mask = np.ones(s_space.shape)
    mask[30:70,30:70] = 0.
    mask = ift.Field.from_global_data(s_space, mask)

    MaskOperator = ift.DiagonalOperator(mask)
    R = ift.GeometryRemover(s_space)
    R = R*MaskOperator
    # R = R*HT
    # R = R * ift.create_harmonic_smoothing_operator((harmonic_space,), 0,
    #                                                response_sigma)
    MeasurementOperator = R

    d_space = MeasurementOperator.target

    Distributor = ift.PowerDistributor(target=h_space, power_space=p_space)
    power = Distributor(ift.exp(0.5*log_p))
    # Creating the mock data
    true_sky = nonlinearity(HT(power*sh))
    noiseless_data = MeasurementOperator(true_sky)
    noise_amplitude = noiseless_data.val.std()*noise_level
    N = ift.ScalingOperator(noise_amplitude**2, d_space)
    n = ift.Field.from_random(
        domain=d_space, random_type='normal',
        std=noise_amplitude, mean=0)
    # Creating the mock data
    d = noiseless_data + n

    m0 = ift.power_synthesize(ift.Field.full(p_space, 1e-7))
    t0 = ift.Field.full(p_space, -4.)
    power0 = Distributor.times(ift.exp(0.5 * t0))

    IC1 = ift.GradientNormController(name="IC1", iteration_limit=100,
                                     tol_abs_gradnorm=1e-3)
    LS = ift.LineSearchStrongWolfe(c2=0.02)
    minimizer = ift.RelaxedNewton(IC1, line_searcher=LS)

    ICI = ift.GradientNormController(iteration_limit=500,
                                     tol_abs_gradnorm=1e-3)
    inverter = ift.ConjugateGradient(controller=ICI)

    for i in range(20):
        power0 = Distributor(ift.exp(0.5*t0))
        map0_energy = ift.library.NonlinearWienerFilterEnergy(
            m0, d, MeasurementOperator, nonlinearity, HT, power0, N, S,
            inverter=inverter)

        # Minimization with chosen minimizer
        map0_energy, convergence = minimizer(map0_energy)
        m0 = map0_energy.position

        # Updating parameters for correlation structure reconstruction
        D0 = map0_energy.curvature

        # Initializing the power energy with updated parameters
        power0_energy = ift.library.NonlinearPowerEnergy(
            position=t0, d=d, N=N, xi=m0, D=D0, ht=HT,
            Instrument=MeasurementOperator, nonlinearity=nonlinearity,
            Distribution=Distributor, sigma=1., samples=2, inverter=inverter)

        power0_energy = minimizer(power0_energy)[0]

        # Setting new power spectrum
        t0 = power0_energy.position

        # break degeneracy between amplitude and excitation by setting
        # excitation monopole to 1
        m0, t0 = adjust_zero_mode(m0, t0)

    plotdict = {"colormap": "Planck-like"}
    ift.plot(true_sky, title="True sky", name="true_sky.png", **plotdict)
    ift.plot(nonlinearity(HT(power0*m0)), title="Reconstructed sky",
             name="reconstructed_sky.png", **plotdict)
    ift.plot(MeasurementOperator.adjoint_times(d), title="Data",
             name="data.png", **plotdict)
