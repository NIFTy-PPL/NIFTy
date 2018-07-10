import nifty5 as ift
import numpy as np


if __name__ == '__main__':
    # FIXME ABOUT THIS CODE
    np.random.seed(41)

    # Set up the position space of the signal
    #
    # # One dimensional regular grid with uniform exposure
    # position_space = ift.RGSpace(1024)
    # exposure = np.ones(position_space.shape)

    # Two-dimensional regular grid with inhomogeneous exposure
    position_space = ift.RGSpace([512, 512])

    # # Sphere with with uniform exposure
    # position_space = ift.HPSpace(128)
    # exposure = ift.Field.full(position_space, 1.)

    # Defining harmonic space and transform
    harmonic_space = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)

    domain = ift.MultiDomain.make({'xi': harmonic_space})
    position = ift.from_random('normal', domain)

    # Define power spectrum and amplitudes
    def sqrtpspec(k):
        return 1. / (20. + k**2)

    p_space = ift.PowerSpace(harmonic_space)
    pd = ift.PowerDistributor(harmonic_space, p_space)
    a = ift.PS_field(p_space, sqrtpspec)
    A = pd(a)

    # Set up a sky model
    xi = ift.Variable(position)['xi']
    logsky_h = xi * A
    logsky = HT(logsky_h)
    sky = ift.PointwisePositiveTanh(logsky)

    GR = ift.GeometryRemover(position_space)
    # Set up instrumental response
    R = GR

    # Generate mock data
    d_space = R.target[0]
    p = R(sky)
    mock_position = ift.from_random('normal', p.position.domain)
    pp = p.at(mock_position).value
    data = np.random.binomial(1, pp.to_global_data().astype(np.float64))
    data = ift.Field.from_global_data(d_space, data)

    # Compute likelihood and Hamiltonian
    position = ift.from_random('normal', p.position.domain)
    likelihood = ift.BernoulliEnergy(p, data)
    ic_cg = ift.GradientNormController(iteration_limit=50)
    ic_newton = ift.GradientNormController(name='Newton', iteration_limit=30,
                                           tol_abs_gradnorm=1e-3)
    minimizer = ift.RelaxedNewton(ic_newton)
    ic_sampling = ift.GradientNormController(iteration_limit=100)

    # Minimize the Hamiltonian
    H = ift.Hamiltonian(likelihood, ic_sampling)
    H = H.makeInvertible(ic_cg)
    # minimizer = ift.SteepestDescent(ic_newton)
    H, convergence = minimizer(H)

    reconstruction = sky.at(H.position).value

    ift.plot(reconstruction, title='reconstruction', name='reconstruction.png')
    ift.plot(GR.adjoint_times(data), title='data', name='data.png')
    ift.plot(sky.at(mock_position).value, title='truth', name='truth.png')
