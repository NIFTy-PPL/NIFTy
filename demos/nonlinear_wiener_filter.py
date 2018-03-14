import nifty4 as ift
from nifty4.library.nonlinearities import Linear, Exponential, Tanh
import numpy as np
np.random.seed(20)

if __name__ == "__main__":

    noise_level = 0.3
    correlation_length = 0.1
    p_spec = lambda k: (1. / (k*correlation_length + 1) ** 4)

    nonlinearity = Tanh()
    #nonlinearity = Linear()
    #nonlinearity = Exponential()

    # Set up position space
    s_space = ift.RGSpace(1024)
    h_space = s_space.get_default_codomain()

    # Define harmonic transformation and associated harmonic space
    HT = ift.HarmonicTransformOperator(h_space, target=s_space)

    S = ift.ScalingOperator(1., h_space)

    # Drawing a sample sh from the prior distribution in harmonic space
    sh = S.draw_sample()

    # Choosing the measurement instrument
    # Instrument = SmoothingOperator(s_space, sigma=0.01)
    mask = np.ones(s_space.shape)
    mask[600:800] = 0.
    mask = ift.Field.from_global_data(s_space, mask)

    R = ift.GeometryRemover(s_space) * ift.DiagonalOperator(mask)

    d_space = R.target

    power = ift.sqrt(ift.create_power_operator(h_space, p_spec).diagonal)

    # Creating the mock data
    true_sky = nonlinearity(HT(power*sh))
    noiseless_data = R(true_sky)
    noise_amplitude = noiseless_data.val.std()*noise_level
    N = ift.ScalingOperator(noise_amplitude**2, d_space)
    n = N.draw_sample()
    # Creating the mock data
    d = noiseless_data + n

    IC1 = ift.GradientNormController(name="IC1", iteration_limit=100,
                                     tol_abs_gradnorm=1e-4)
    LS = ift.LineSearchStrongWolfe(c2=0.02)
    minimizer = ift.RelaxedNewton(IC1, line_searcher=LS)

    ICI = ift.GradientNormController(iteration_limit=2000,
                                     tol_abs_gradnorm=1e-3)
    inverter = ift.ConjugateGradient(controller=ICI)

    # initial guess
    m = ift.Field.full(h_space, 1e-7)
    map_energy = ift.library.NonlinearWienerFilterEnergy(
        m, d, R, nonlinearity, HT, power, N, S, inverter=inverter)

    # Minimization with chosen minimizer
    map_energy, convergence = minimizer(map_energy)
    m = map_energy.position

    recsky = nonlinearity(HT(power*m))
    data = R.adjoint_times(d)
    lo = np.min([true_sky.min(), recsky.min(), data.min()])
    hi = np.max([true_sky.max(), recsky.max(), data.max()])
    plotdict = {"colormap": "Planck-like", "ymin": lo, "ymax": hi}
    ift.plot(true_sky, name="true_sky.png", **plotdict)
    ift.plot(recsky, name="reconstructed_sky.png", **plotdict)
    ift.plot(data, name="data.png", **plotdict)
