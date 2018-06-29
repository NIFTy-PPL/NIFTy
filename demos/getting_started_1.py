import nifty5 as ift
import numpy as np


def make_chess_mask():
    mask = np.ones(position_space.shape)
    for i in range(4):
        for j in range(4):
            if (i+j)%2 == 0:
<<<<<<< HEAD
                mask[i*512/4:(i+1)*512/4, j*512/4:(j+1)*512/4] = 0
=======
                mask[i*128//4:(i+1)*128//4, j*128//4:(j+1)*128//4] = 0
>>>>>>> 2b5d58fda1926161b86883a9e639f969c3c7e4fb
    return mask


def make_random_mask():
    mask = ift.from_random('pm1',position_space)
    mask = (mask+1)/2
    return mask.val


if __name__ == '__main__':
    ## describtion of the tutorial ###

    # Choose problem geometry and masking

    # # One dimensional regular grid
    # position_space = ift.RGSpace([1024])
    # mask = np.ones(position_space.shape)

<<<<<<< HEAD
    # # Two dimensional regular grid with chess mask
    position_space = ift.RGSpace([512,512])
=======
    # Two dimensional regular grid with chess mask
    position_space = ift.RGSpace([128,128])
>>>>>>> 2b5d58fda1926161b86883a9e639f969c3c7e4fb
    mask = make_chess_mask()

    # # Sphere with half of its locations randomly masked
    # position_space = ift.HPSpace(128)
    # mask = make_random_mask()

    # set up corresponding harmonic transform and space
    harmonic_space = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)

    # set correlation structure with a power spectrum and build prior correlation covariance
    def power_spectrum(k):
        return 100. / (20.+k**3)
    power_space = ift.PowerSpace(harmonic_space)
    PD = ift.PowerDistributor(harmonic_space, power_space)
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))

    S = ift.DiagonalOperator(prior_correlation_structure)

    # build instrument response consisting of a discretization, mask and harmonic transformaion
    GR = ift.GeometryRemover(position_space)
    mask = ift.Field(position_space,val=mask)
    Mask = ift.DiagonalOperator(mask)
    R = GR * Mask * HT

    data_space = GR.target

    # setting the noise covariance
    noise = 5.
    N = ift.ScalingOperator(noise, data_space)

    # creating mock data
    MOCK_SIGNAL = S.draw_sample()
    MOCK_NOISE = N.draw_sample()
    data = R(MOCK_SIGNAL) + MOCK_NOISE

    # building propagator D and information source j
    j = R.adjoint_times(N.inverse_times(data))
    D_inv = R.adjoint * N.inverse * R + S.inverse
    # make it invertible
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv,IC,approximation=S.inverse).inverse

    # WIENER FILTER
    m = D(j)

    ##PLOTTING
    #Truth, data, reconstruction, residuals
