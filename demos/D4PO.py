# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import nifty5 as ift
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def get_RG_instrument():
    pointings = 1000*np.random.binomial(1,0.001,position_space.shape)
    psf = np.zeros_like(pointings)
    psf[0,0] = 1.

    sigmas_sensitivity = np.arange(0.1,0.5,0.4/energy_space.shape[0])
    sigmas_psf = np.arange(0.005,0.0125,0.0075/energy_space.shape[0])
    total_psf = np.empty(htp.domain.shape)
    total_psf_sens = np.empty(htp.domain.shape)

    sensitivity = np.empty(total_space.shape)
    sensitivity.T[:]=pointings
    kernel = hp_space.get_k_length_array()

    for i in range(energy_space.shape[0]):
        smoother = hp_space.get_fft_smoothing_kernel_function(sigmas_psf[i])
        smoother_sens = hp_space.get_fft_smoothing_kernel_function(sigmas_sensitivity[i])

        total_psf.T[i] = np.round(smoother(kernel).val,20)
        total_psf_sens.T[i] = np.round(smoother_sens(kernel).val,20)


    sensitivity = ift.Field.from_global_data(total_space, sensitivity)
    total_psf = ift.Field.from_global_data(htp.domain, total_psf)
    total_psf_sens = ift.Field.from_global_data(htp.domain, total_psf_sens)


    HT = ift.HartleyOperator(htp.domain,position_space,space=0)
    K = ift.DiagonalOperator(total_psf)
    KK = ift.DiagonalOperator(total_psf_sens)
    sensitivity = (HT @ KK @ HT.inverse)(sensitivity)
    E = ift.DiagonalOperator(sensitivity)
    PSF = HT @ K @ HT.inverse
    G = ift.GeometryRemover(total_space)
    return G @ E @ PSF


def get_PsfOp():
    # kernel = get_kernel()
    h_space_spatial = position_space.get_default_codomain()

    sigmas_psf = np.arange(0.02,0.04,0.02/energy_space.shape[0])
    total_psf = np.empty(htp.domain.shape)
    kernel = hp_space.get_k_length_array()
    for i in range(energy_space.shape[0]):
        smoother = hp_space.get_fft_smoothing_kernel_function(sigmas_psf[i])
        total_psf.T[i] = np.round(smoother(kernel).val,20)
    total_psf = ift.Field.from_global_data(htp.domain, total_psf)
    K = ift.DiagonalOperator(total_psf)

    # gau = h_space_spatial.get_fft_smoothing_kernel_function(0.01)
    # k_l_a = h_space_spatial.get_k_length_array()
    # kernel = gau(k_l_a).val
    scaling = 4*np.pi/12/position_space.nside**2
    he_space = ift.DomainTuple.make((h_space_spatial,energy_space))
    Scaling = ift.ScalingOperator(scaling,he_space)

    HT = ift.HarmonicTransformOperator(he_space, target=position_space, space=0)
    PI = ift.ScalingOperator(4*np.pi, HT.target)
    # multi_kernel = np.empty(he_space.shape)
    # multi_kernel.T[:] = kernel
    # multi_kernel = ift.Field(he_space,val=multi_kernel)
    # Kernel = ift.DiagonalOperator(multi_kernel)
    return PI @ HT @ K @ Scaling @ HT.adjoint




class ReverseOuterProduct(ift.LinearOperator):
    """Performs the pointwise outer product of two fields.

    Parameters
    ---------
    field: Field,
    domain: DomainTuple, the domain of the input field
    ---------
    """

    def __init__(self, field, domain):

        self._domain = domain
        self._field = field
        self._target = ift.DomainTuple.make(
            tuple(sub_d for sub_d in  domain._dom + field.domain._dom))

        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return ift.Field.from_global_data(
                self._target, np.multiply.outer(
                     x.to_global_data(), self._field.to_global_data()))
        axes = len(self._field.shape)
        return ift.Field.from_global_data(
            self._domain, np.tensordot(
                 x.to_global_data(), self._field.to_global_data(),  axes))


if __name__ == '__main__':
    # FIXME description of the tutorial
    np.random.seed(41)
    np.seterr(all='raise')
    # position_space = ift.RGSpace([128,128])
    position_space = ift.HPSpace(64)
    energy_space = ift.RGSpace([4])


    total_space = ift.DomainTuple.make([position_space,energy_space])

    # Setting up an amplitude models
    A_p = ift.AmplitudeOperator(position_space, 64, 3, 0.4, -3., 1, 3, 1.,keys=['tau_p','phi_p'])
    A_e = ift.AmplitudeOperator(energy_space, 8, 3, 0.4, -2., 1, 1, 0.,keys=['tau_e','phi_e'])
    A_eu = ift.AmplitudeOperator(energy_space, 8, 0.3, 0.4, -4., 1, -5, 1.,keys=['tau_eu','phi_eu'],zero_mode=False)

    correlated_field = ift.MfCorrelatedField(position_space,energy_space,A_p,A_e,)
    # Building the model for a correlated signal
    hp_space = position_space.get_default_codomain()
    he_space = energy_space.get_default_codomain()
    hpe_space = ift.DomainTuple.make([hp_space,he_space])
    hte = ift.HarmonicTransformOperator(hpe_space, space=1)
    htp = ift.HarmonicTransformOperator(hte.target,target = position_space,space=0)
    ht = htp @ hte

    # EXPERIMENTAL
    peu_space = A_eu.target[0]

    Peu = ift.PowerDistributor(he_space, peu_space)

    one_p = ift.full(hp_space,val=1.)
    one_u = ift.full(energy_space,val=1.)
    OP_e = ift.OuterProduct(one_p,ift.DomainTuple.make(he_space))
    OP_u = ReverseOuterProduct(one_u,ift.DomainTuple.make(position_space))

    PA_eu = OP_e(Peu(A_eu))
    #
    scale = ift.ScalingOperator(1e-16,ht.target)
    off = ift.OffsetOperator(ift.full(ht.target,0.))
    xi_u = ift.FieldAdapter(hpe_space, "xi_u")
    u_spectra = (off((ht(PA_eu * xi_u)))).absolute()
    u = ift.InverseGammaOperator(position_space,1.0,1e-3)(ift.FieldAdapter(position_space,'u'))
    points = u_spectra * OP_u(u)

    signal = ift.exp(correlated_field) + points

    # Building the Line of Sight response
    G = ift.GeometryRemover(signal.target)
    # exposure = get_3D_exposure()
    # exposure = ift.Field.from_global_data(signal.target,exposure)
    # E = ift.DiagonalOperator(exposure)
    # PSF = get_PsfOp()

    R = G #@ PSF
    # R =  get_RG_instrument()
    # build signal response model and model likelihood
    signal_response = R(signal).clip(min=1e-15)
    # specify noise
    data_space = R.target

    # generate mock data
    MOCK_POSITION = ift.from_random('normal', signal_response.domain)
    rate = signal_response(MOCK_POSITION)
    data = np.random.poisson(np.abs(rate.to_global_data().astype(np.float64)))
    data = ift.Field.from_global_data(data_space, data)


    # set up model likelihood
    likelihood = ift.PoissonianEnergy(data)(signal_response)

    # set up minimization and inversion schemes
    ic_sampling = ift.GradientNormController(iteration_limit=50)
    ic_newton = ift.GradInfNormController(
        name='Newton', tol=1e-7, iteration_limit=10)
    minimizer = ift.NewtonCG(ic_newton)
    # build model Hamiltonian
    H = ift.Hamiltonian(likelihood, ic_sampling)

    INITIAL_POSITION = 0.1*ift.from_random('normal', H.domain)
    position = INITIAL_POSITION

    plo = ift.Plot()
    for i in range(9):
        plo.add(signal(ift.from_random('normal',signal.domain)))
    plo.output(name='prior_samples.png', nx=3,ny=3)

    # number of samples used to estimate the KL
    N_samples = 3
    for i in range(5):
        KL = ift.KL_Energy(position, H, N_samples, gen_mirrored_samples=True)
        KL, convergence = minimizer(KL)
        position = KL.position

        plo = ift.Plot()
        plo.add(G.adjoint_times(data) , title='data')
        plo.add(G.adjoint_times(signal_response(MOCK_POSITION))+, title='signal response')
        plo.add(signal(MOCK_POSITION), title='true signal')
        plo.add(signal(position), title='signal')
        plo.add(ift.exp(correlated_field.force(MOCK_POSITION)), title='truediffuse')
        plo.add(ift.exp(correlated_field.force(position)), title='diffuse')

        plo.add(points.force(position), title='points')

        plo.add(signal(position + KL.samples[0]), title='sample')

        plo.output(name='signal.pdf', nx=2, ny=4, xsize=15, ysize=20)

