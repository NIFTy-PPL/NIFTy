import nifty2go as ift
from nifty2go.library.nonlinearities import Exponential
import numpy as np
np.random.seed(42)


def adjust_zero_mode(m0, t0):
    zero_position = len(m0.shape)*(0,)
    zero_mode = m0.val[zero_position]
    m0.val[zero_position] = zero_mode / abs(zero_mode)
    t0.val[0] += 2 * np.log(abs(zero_mode))
    return m0, t0


if __name__ == "__main__":

    noise_level = 1.
    p_spec = (lambda k: (1. / (k + 1) ** 2))

    # nonlinearity = Linear()
    nonlinearity = Exponential()
    # Set up position space
    # s_space = ift.RGSpace([1024])
    s_space = ift.HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    FFT = ift.FFTOperator(s_space)
    h_space = FFT.target[0]

    # Setting up power space
    p_space = ift.PowerSpace(h_space,
                             binbounds=ift.PowerSpace.useful_binbounds(
                                 h_space, logarithmic=True))
    s_spec = ift.Field(p_space, val=1.)
    # Choosing the prior correlation structure and defining
    # correlation operator
    p = ift.PS_field(p_space, p_spec)
    log_p = ift.log(p)
    S = ift.create_power_operator(h_space, power_spectrum=s_spec)

    # Drawing a sample sh from the prior distribution in harmonic space
    sp = ift.Field(p_space, val=s_spec)
    sh = ift.power_synthesize(sp)

    # Choosing the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.01)
    mask = np.ones(s_space.shape)
    mask[6000:8000] = 0.
    mask = ift.Field(s_space, val=ift.dobj.from_global_data(mask))

    MaskOperator = ift.DiagonalOperator(mask)
    InstrumentResponse = ift.ResponseOperator(s_space, sigma=[0.0],
                                              exposure=[1.0])
    MeasurementOperator = InstrumentResponse*MaskOperator

    d_space = MeasurementOperator.target

    noise_covariance = ift.Field(d_space, val=noise_level**2).weight()
    N = ift.DiagonalOperator(noise_covariance)
    n = ift.Field.from_random(domain=d_space, random_type='normal',
                              std=noise_level)
    Projection = ift.PowerProjectionOperator(domain=h_space,
                                             power_space=p_space)
    power = Projection.adjoint_times(ift.exp(0.5*log_p))
    # Creating the mock data
    true_sky = nonlinearity(FFT.adjoint_times(power*sh))
    d = MeasurementOperator(true_sky) + n

    m0 = ift.power_synthesize(ift.Field(p_space, val=1e-7))
    t0 = ift.Field(p_space, val=-4.)
    power0 = Projection.adjoint_times(ift.exp(0.5 * t0))

    IC1 = ift.GradientNormController(verbose=True, iteration_limit=100,
                                     tol_abs_gradnorm=1e-3)
    LS = ift.LineSearchStrongWolfe(c2=0.02)
    minimizer = ift.RelaxedNewton(IC1, line_searcher=LS)

    ICI = ift.GradientNormController(verbose=False, name="ICI",
                                     iteration_limit=500,
                                     tol_abs_gradnorm=1e-3)
    inverter = ift.ConjugateGradient(controller=ICI)

    for i in range(20):
        power0 = Projection.adjoint_times(ift.exp(0.5*t0))
        map0_energy = ift.library.NonlinearWienerFilterEnergy(
            m0, d, MeasurementOperator, nonlinearity, FFT, power0, N, S,
            inverter=inverter)

        # Minimization with chosen minimizer
        map0_energy, convergence = minimizer(map0_energy)
        m0 = map0_energy.position

        # Updating parameters for correlation structure reconstruction
        D0 = map0_energy.curvature

        # Initializing the power energy with updated parameters
        power0_energy = ift.library.NonlinearPowerEnergy(
            position=t0, d=d, N=N, m=m0, D=D0, FFT=FFT,
            Instrument=MeasurementOperator, nonlinearity=nonlinearity,
            Projection=Projection, sigma=1., samples=2, inverter=inverter)

        power0_energy = minimizer(power0_energy)[0]

        # Setting new power spectrum
        t0 = power0_energy.position

        # break degeneracy between amplitude and excitation by setting
        # excitation monopole to 1
        m0, t0 = adjust_zero_mode(m0, t0)

    ift.plotting.plot(true_sky)
    ift.plotting.plot(nonlinearity(FFT.adjoint_times(power0*m0)),
                      title='reconstructed_sky')
    ift.plotting.plot(MeasurementOperator.adjoint_times(d))
