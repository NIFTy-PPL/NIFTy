import nifty5 as ift
import numpy as np


def get_2D_exposure():
    x_shape, y_shape = position_space.shape

    exposure = np.ones(position_space.shape)
    exposure[x_shape//3:x_shape//2, :] *= 2.
    exposure[x_shape*4//5:x_shape, :] *= .1
    exposure[x_shape//2:x_shape*3//2, :] *= 3.
    exposure[:, x_shape//3:x_shape//2] *= 2.
    exposure[:, x_shape*4//5:x_shape] *= .1
    exposure[:, x_shape//2:x_shape*3//2] *= 3.

    exposure = ift.Field.from_global_data(position_space, exposure)
    return exposure


if __name__ == '__main__':
    # ABOUT THIS CODE
    np.random.seed(41)

    # Set up the position space of the signal
    #
    # # One dimensional regular grid with uniform exposure
    # position_space = ift.RGSpace(1024)
    # exposure = np.ones(position_space.shape)

    # Two-dimensional regular grid with inhomogeneous exposure
    position_space = ift.RGSpace([512, 512])
    exposure = get_2D_exposure()

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
    sky = ift.PointwiseExponential(logsky)

    M = ift.DiagonalOperator(exposure)
    GR = ift.GeometryRemover(position_space)
    # Set up instrumental response
    R = GR * M

    # Generate mock data
    d_space = R.target[0]
    lamb = R(sky)
    mock_position = ift.from_random('normal', lamb.position.domain)
    data = lamb.at(mock_position).value
    data = np.random.poisson(data.to_global_data().astype(np.float64))
    data = ift.Field.from_global_data(d_space, data)

    # Compute likelihood and Hamiltonian
    position = ift.from_random('normal', lamb.position.domain)
    likelihood = ift.PoissonianEnergy(lamb, data)
    ic_cg = ift.GradientNormController(iteration_limit=50)
    ic_newton = ift.GradientNormController(name='Newton', iteration_limit=50,
                                           tol_abs_gradnorm=1e-3)
    minimizer = ift.RelaxedNewton(ic_newton)

    # Minimize the Hamiltonian
    H = ift.Hamiltonian(likelihood)
    H = H.makeInvertible(ic_cg)
    H, convergence = minimizer(H)

    # Plot results
    result_sky = sky.at(H.position).value
    # PLOTTING
