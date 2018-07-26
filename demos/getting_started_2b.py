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

def myexp(lin):
    if isinstance(lin, ift.Linearization):
        tmp = ift.exp(lin.val)
        return ift.Linearization(tmp, ift.makeOp(tmp)*lin.jac)
    return ift.exp(lin)

def mylog(lin):
    if isinstance(lin, ift.Linearization):
        tmp = ift.log(lin.val)
        return ift.Linearization(tmp, ift.makeOp(1./lin.val)*lin.jac)
    return ift.log(lin)

class GaussianEnergy2(ift.Operator):
    def __init__(self, mean=None, covariance=None):
        super(GaussianEnergy2, self).__init__()
        self._mean = mean
        self._icov = None if covariance is None else covariance.inverse

    def __call__(self, x):
        residual = x if self._mean is None else x-self._mean
        icovres = residual if self._icov is None else self._icov(residual)
        res = .5 * (residual*icovres).sum()
        metric = ift.SandwichOperator.make(x.jac, self._icov)
        return ift.Linearization(res.val, res.jac, metric)

class PoissonianEnergy2(ift.Operator):
    def __init__(self, op, d):
        super(PoissonianEnergy2, self).__init__()
        self._op = op
        self._d = d

    def __call__(self, x):
        x = self._op(x)
        res = (x - self._d*mylog(x)).sum()
        metric = ift.SandwichOperator.make(x.jac, ift.makeOp(1./x.val))
        return ift.Linearization(res.val, res.jac, metric)

class MyHamiltonian(ift.Operator):
    def __init__(self, lh):
        super(MyHamiltonian, self).__init__()
        self._lh = lh
        self._prior = GaussianEnergy2()

    def __call__(self, x):
        return self._lh(x) + self._prior(x)

class EnergyAdapter(ift.Energy):
    def __init__(self, position, op):
        super(EnergyAdapter, self).__init__(position)
        self._op = op
        pvar = ift.Linearization.make_var(position)
        self._res = op(pvar)

    def at(self, position):
        return EnergyAdapter(position, self._op)

    @property
    def value(self):
        return self._res.val.local_data[()]

    @property
    def gradient(self):
        return self._res.jac.adjoint_times(ift.full(self._res.jac.target, 1.))

    @property
    def metric(self):
        return self._res.metric


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
    # FIXME description of the tutorial
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

    domain = ift.DomainTuple.make(harmonic_space)
    position = ift.from_random('normal', domain)

    # Define power spectrum and amplitudes
    def sqrtpspec(k):
        return 1. / (20. + k**2)

    p_space = ift.PowerSpace(harmonic_space)
    pd = ift.PowerDistributor(harmonic_space, p_space)
    a = ift.PS_field(p_space, sqrtpspec)
    A = pd(a)

    # Set up a sky model
    sky = lambda inp: myexp(HT(inp*A))

    M = ift.DiagonalOperator(exposure)
    GR = ift.GeometryRemover(position_space)
    # Set up instrumental response
    R = GR * M

    # Generate mock data
    d_space = R.target[0]
    lamb = lambda inp: R(sky(inp))
    mock_position = ift.from_random('normal', domain)
    data = lamb(mock_position)
    data = np.random.poisson(data.to_global_data().astype(np.float64))
    data = ift.Field.from_global_data(d_space, data)

    # Compute likelihood and Hamiltonian
    likelihood = PoissonianEnergy2(lamb, data)
    ic_cg = ift.GradientNormController(iteration_limit=50)
    ic_newton = ift.GradientNormController(name='Newton', iteration_limit=50,
                                           tol_abs_gradnorm=1e-3)
    minimizer = ift.RelaxedNewton(ic_newton)

    # Minimize the Hamiltonian
    H = MyHamiltonian(likelihood)
    H = EnergyAdapter(position, H)
    #ift.extra.check_value_gradient_consistency(H)
    H = H.make_invertible(ic_cg)
    H, convergence = minimizer(H)

    # Plot results
    result_sky = sky(H.position)
    ift.plot(result_sky)
    ift.plot_finish()
    # FIXME PLOTTING
