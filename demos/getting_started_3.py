import nifty5 as ift
import numpy as np


def get_random_LOS(n_los):
    starts = list(np.random.uniform(0, 1, (n_los, 2)).T)
    ends = list(np.random.uniform(0, 1, (n_los, 2)).T)

    return starts, ends


if __name__ == '__main__':
    # ## ABOUT THIS TUTORIAL
    np.random.seed(42)
    position_space = ift.RGSpace([128, 128])

    # Setting up an amplitude model
    A, amplitude_internals = ift.make_amplitude_model(
        position_space, 16, 1, 10, -4., 1, 0., 1.)

    # Building the model for a correlated signal
    harmonic_space = position_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(harmonic_space, position_space)
    power_space = A.value.domain[0]
    power_distributor = ift.PowerDistributor(harmonic_space, power_space)
    position = {}
    position['xi'] = ift.Field.from_random('normal', harmonic_space)
    position = ift.MultiField(position)

    xi = ift.Variable(position)['xi']
    Amp = power_distributor(A)
    correlated_field_h = Amp * xi
    correlated_field = ht(correlated_field_h)
    # alternatively to the block above one can do:
    # correlated_field,_ = ift.make_correlated_field(position_space, A)

    # apply some nonlinearity
    signal = ift.PointwisePositiveTanh(correlated_field)
    # Building the Line of Sight response
    LOS_starts, LOS_ends = get_random_LOS(100)
    R = ift.LOSResponse(position_space, starts=LOS_starts,
                        ends=LOS_ends)
    # build signal response model and model likelihood
    signal_response = R(signal)
    # specify noise
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # generate mock data
    MOCK_POSITION = ift.from_random('normal', signal.position.domain)
    data = signal_response.at(MOCK_POSITION).value + N.draw_sample()

    # set up model likelihood
    likelihood = ift.GaussianEnergy(signal_response, mean=data, covariance=N)

    # set up minimization and inversion schemes
    ic_cg = ift.GradientNormController(iteration_limit=10)
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradientNormController(name='Newton', iteration_limit=100)
    minimizer = ift.RelaxedNewton(ic_newton)

    # build model Hamiltonian
    H = ift.Hamiltonian(likelihood, ic_sampling)

    INITIAL_POSITION = ift.from_random('normal', H.position.domain)
    position = INITIAL_POSITION

    ift.plot(signal.at(MOCK_POSITION).value, name='truth.pdf')
    ift.plot(R.adjoint_times(data), name='data.pdf')
    ift.plot([A.at(MOCK_POSITION).value], name='power.pdf')

    # number of samples used to estimate the KL
    N_samples = 20
    for i in range(5):
        H = H.at(position)
        samples = [H.metric.draw_sample(from_inverse=True)
                   for _ in range(N_samples)]

        KL = ift.SampledKullbachLeiblerDivergence(H, samples)
        KL = KL.makeInvertible(ic_cg)
        KL, convergence = minimizer(KL)
        position = KL.position

        ift.plot(signal.at(position).value, name='reconstruction.pdf')

        ift.plot([A.at(position).value, A.at(MOCK_POSITION).value],
                 name='power.pdf')

    avrg = 0.
    va = 0.
    powers = []
    for sample in samples:
        sam = signal.at(sample + position).value
        powers.append(A.at(sample+position).value)
        avrg += sam
        va += sam**2

    avrg /= len(samples)
    va /= len(samples)
    va -= avrg**2
    std = ift.sqrt(va)
    ift.plot(avrg, name='avrg.pdf')
    ift.plot(std, name='std.pdf')
    ift.plot([A.at(position).value, A.at(MOCK_POSITION).value]+powers,
             name='power.pdf')
