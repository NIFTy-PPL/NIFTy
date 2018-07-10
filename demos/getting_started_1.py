import nifty5 as ift
import numpy as np


def make_chess_mask(position_space):
    mask = np.ones(position_space.shape)
    for i in range(4):
        for j in range(4):
            if (i+j) % 2 == 0:
                mask[i*128//4:(i+1)*128//4, j*128//4:(j+1)*128//4] = 0
    return mask


def make_random_mask():
    mask = ift.from_random('pm1', position_space)
    mask = (mask+1)/2
    return mask.to_global_data()


if __name__ == '__main__':
    np.random.seed(42)
    # FIXME description of the tutorial

    # Choose problem geometry and masking

    # One dimensional regular grid
    position_space = ift.RGSpace([1024])
    mask = np.ones(position_space.shape)

    # # Two dimensional regular grid with chess mask
    # position_space = ift.RGSpace([128, 128])
    # mask = make_chess_mask(position_space)

    # # Sphere with half of its locations randomly masked
    # position_space = ift.HPSpace(128)
    # mask = make_random_mask()

    harmonic_space = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)

    # Set correlation structure with a power spectrum and build
    # prior correlation covariance
    def power_spectrum(k):
        return 100. / (20.+k**3)
    power_space = ift.PowerSpace(harmonic_space)
    PD = ift.PowerDistributor(harmonic_space, power_space)
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))

    S = ift.DiagonalOperator(prior_correlation_structure)

    # Build instrument response consisting of a discretization, mask
    # and harmonic transformaion
    GR = ift.GeometryRemover(position_space)
    mask = ift.Field.from_global_data(position_space, mask)
    Mask = ift.DiagonalOperator(mask)
    R = GR * Mask * HT

    data_space = GR.target

    # Set the noise covariance
    noise = 5.
    N = ift.ScalingOperator(noise, data_space)

    # Create mock data
    MOCK_SIGNAL = S.draw_sample()
    MOCK_NOISE = N.draw_sample()
    data = R(MOCK_SIGNAL) + MOCK_NOISE

    # Build propagator D and information source j
    j = R.adjoint_times(N.inverse_times(data))
    D_inv = R.adjoint * N.inverse * R + S.inverse
    # Make it invertible
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse

    # WIENER FILTER
    m = D(j)

    # PLOTTING
    rg = isinstance(position_space, ift.RGSpace)
    if rg and len(position_space.shape) == 1:
        ift.plot([HT(MOCK_SIGNAL), GR.adjoint(data), HT(m)],
                 label=['Mock signal', 'Data', 'Reconstruction'],
                 alpha=[1, .3, 1],
                 name='getting_started_1.png')
    else:
        ift.plot(HT(MOCK_SIGNAL), title='Mock Signal', name='mock_signal.png')
        ift.plot((GR*Mask).adjoint(data), title='Data', name='data.png')
        ift.plot(HT(m), title='Reconstruction', name='reconstruction.png')
    ift.plot(HT(m-MOCK_SIGNAL), name='residuals.png')
