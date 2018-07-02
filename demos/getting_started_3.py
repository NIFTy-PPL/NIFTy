import nifty5 as ift
from nifty5.library.los_response import LOSResponse
from nifty5.library.apply_data import ApplyData
from nifty5.library.unit_log_gauss import UnitLogGauss
from nifty5.library.amplitude_model import make_amplitude_model
from nifty5.library.smooth_sky import make_correlated_field
import numpy as np

# def get_radial_LOS(lines=100, rotations=100, scale=1/400.):
#     def make_los(n=10, angle=0, d=1 / 40.):
#         starts_list = []
#         ends_list = []
#         for i in xrange(n):
#             starts_list += [[-(110.) * d + 0.5, d * 0 + 0.5]]
#
#             ends_list += [[(190.) * d + 0.5, (-57.4 + (114.8 * i) / n) * d + 0.5]]
#         starts_list = np.array(starts_list)
#         ends_list = np.array(ends_list)
#
#         rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
#                                [np.sin(angle), np.cos(angle)]])
#         starts_list = rot_matrix.dot(starts_list.T - 0.5).T + 0.5
#         ends_list = rot_matrix.dot(ends_list.T - 0.5).T + 0.5
#
#         return (starts_list, ends_list)
#
#     rotation_angle = np.pi*2
#     temp_coords = (np.empty((0, 2)), np.empty((0, 2)))
#     for alpha in [-rotation_angle/rotations*j for j in xrange(rotations)]:
#         temp = make_los(n=lines, angle=alpha, d = scale)
#         temp_coords = np.concatenate([temp_coords, temp], axis=1)
#
#     starts = list(temp_coords[0].T)
#     ends = list(temp_coords[1].T)
#     return starts, ends

def get_random_LOS(n_los):
    starts = list(np.random.uniform(0,1,(n_los,2)).T)
    ends = list(np.random.uniform(0,1,(n_los,2)).T)

    return starts, ends


if __name__ == '__main__':
    np.random.seed(41)
    position_space = ift.RGSpace([128,128])

    A, __ = make_amplitude_model(position_space,16, 1, 10, -4., 1, 0., 1.)
    log_signal, _ = make_correlated_field(position_space,A)
    signal = ift.PointwisePositiveTanh(log_signal)
    # LOS_starts, LOS_ends = get_radial_LOS()
    LOS_starts, LOS_ends = get_random_LOS(100)
    R = LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    # R = ift.GeometryRemover(position_space)
    data_space = R.target
    signal_response = R(signal)
    noise = .001
    N = ift.ScalingOperator(noise,data_space)
    MOCK_POSITION = ift.from_random('normal', signal.position.domain)

    data = signal_response.at(MOCK_POSITION).value + N.draw_sample()
    NWR = ApplyData(data,ift.Field(data_space,val=noise), signal_response)
    ic_cg = ift.GradientNormController(iteration_limit=10)
    ic_sampling = ift.GradientNormController(iteration_limit=100)

    likelihood = UnitLogGauss(NWR,ic_cg)
    H = ift.Hamiltonian(likelihood,ic_cg,iteration_controller_sampling=ic_sampling)
    N_samples = 20
    IC2 = ift.GradientNormController(name='Newton', iteration_limit=100)
    minimizer = ift.RelaxedNewton(IC2)
    INITIAL_POSITION = ift.from_random('normal',H.position.domain)
    position = INITIAL_POSITION
    ift.plot(signal.at(MOCK_POSITION).value,name='truth.pdf')
    ift.plot(R.adjoint_times(data),name='data.pdf')
    ift.plot([ A.at(MOCK_POSITION).value], name='power.pdf')

    # H, convergence = minimizer(H)
    # position = H.position
    for i in range(5):
        H = H.at(position)
        samples = [H.curvature.draw_sample(from_inverse=True) for _ in range(N_samples)]

        KL = ift.SampledKullbachLeiblerDivergence(H, samples, ic_cg)
        KL, convergence = minimizer(KL)
        position = KL.position

        ift.plot(signal.at(position).value,name='reconstruction.pdf')

        ift.plot([A.at(position).value, A.at(MOCK_POSITION).value],name='power.pdf')
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
    ift.plot([A.at(position).value, A.at(MOCK_POSITION).value]+powers, name='power.pdf')











