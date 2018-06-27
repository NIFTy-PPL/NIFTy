import nifty5 as ift
import numpy as np
from global_newton.models_other.apply_data import ApplyData
from global_newton.models_energy.hamiltonian import Hamiltonian
from nifty5.library.unit_log_gauss import UnitLogGauss
if __name__ == '__main__':
    # s_space = ift.RGSpace([1024])
    s_space = ift.RGSpace([128,128])
    # s_space = ift.HPSpace(64)

    h_space = s_space.get_default_codomain()
    total_domain = ift.MultiDomain.make({'xi': h_space})
    HT = ift.HarmonicTransformOperator(h_space, s_space)

    def sqrtpspec(k):
        return 16. / (20.+k**2)

    GR = ift.GeometryRemover(s_space)

    d_space = GR.target
    B = ift.FFTSmoothingOperator(s_space,0.1)
    mask = np.ones(s_space.shape)
    mask[64:89,76:100] = 0.
    mask = ift.Field(s_space,val=mask)
    Mask = ift.DiagonalOperator(mask)
    R = GR * Mask * B
    noise = 1.
    N = ift.ScalingOperator(noise, d_space)

    p_space = ift.PowerSpace(h_space)
    pd = ift.PowerDistributor(h_space, p_space)
    position = ift.from_random('normal', total_domain)
    xi = ift.Variable(position)['xi']
    a = ift.Constant(position, ift.PS_field(p_space, sqrtpspec))
    A = pd(a)
    s_h = A * xi
    s = HT(s_h)
    Rs = R(s)



    MOCK_POSITION = ift.from_random('normal',total_domain)
    data = Rs.at(MOCK_POSITION).value + N.draw_sample()

    NWR = ApplyData(data, ift.Field(d_space,val=noise), Rs)

    INITIAL_POSITION = ift.from_random('normal',total_domain)
    likelihood = UnitLogGauss(INITIAL_POSITION, NWR)

    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    inverter = ift.ConjugateGradient(controller=IC)
    IC2 = ift.GradientNormController(name='Newton', iteration_limit=15)
    minimizer = ift.RelaxedNewton(IC2)


    H = Hamiltonian(likelihood, inverter)
    H, convergence = minimizer(H)
    result = s.at(H.position).value


