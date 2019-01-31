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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..minimization.energy_adapter import EnergyAdapter
from ..multi_field import MultiField
from ..operators.distributors import PowerDistributor
from ..operators.energy_operators import (StandardHamiltonian,
                                          InverseGammaLikelihood)
from ..operators.scaling_operator import ScalingOperator
from ..operators.simple_linear_operators import ducktape


def make_adjust_variances(a,
                          xi,
                          position,
                          samples=[],
                          scaling=None,
                          ic_samp=None):
    """Creates a Hamiltonian for constant likelihood optimizations.

    Constructs a Hamiltonian to solve constant likelihood optimizations of the
    form phi = a * xi under the constraint that phi remains constant.

    FIXME xi is white.

    Parameters
    ----------
    a : Operator
        Gives the amplitude when evaluated at a position.
    xi : Operator
        Gives the excitation when evaluated at a position.
    position : Field, MultiField
        Position of the entire problem.
    samples : Field, MultiField
        Residual samples of the whole problem.
    scaling : Float
        Optional rescaling of the Likelihood.
    ic_samp : Controller
        Iteration Controller for Hamiltonian.

    Returns
    -------
    StandardHamiltonian
        A Hamiltonian that can be used for further minimization.
    """

    d = a*xi
    d = (d.conjugate()*d).real
    n = len(samples)
    if n > 0:
        d_eval = 0.
        for i in range(n):
            d_eval = d_eval + d.force(position + samples[i])
        d_eval = d_eval/n
    else:
        d_eval = d.force(position)

    x = (a.conjugate()*a).real
    if scaling is not None:
        x = ScalingOperator(scaling, x.target)(x)

    return StandardHamiltonian(InverseGammaLikelihood(d_eval/2.)(x),
                               ic_samp=ic_samp)


def do_adjust_variances(position,
                        amplitude_operator,
                        minimizer,
                        xi_key='xi',
                        samples=[]):
    '''
    FIXME
    '''

    h_space = position[xi_key].domain[0]
    pd = PowerDistributor(h_space, amplitude_operator.target[0])
    a = pd(amplitude_operator)
    xi = ducktape(None, position.domain, xi_key)

    ham = make_adjust_variances(a, xi, position, samples=samples)

    # Minimize
    e = EnergyAdapter(
        position.extract(a.domain), ham, constants=[], want_metric=True)
    e, _ = minimizer(e)

    # Update position
    s_h_old = (a*xi).force(position)

    position = position.to_dict()
    position[xi_key] = s_h_old/a(e.position)
    position = MultiField.from_dict(position)
    position = MultiField.union([position, e.position])

    s_h_new = (a*xi).force(position)

    import numpy as np
    # TODO Move this into the tests
    np.testing.assert_allclose(s_h_new.to_global_data(),
                               s_h_old.to_global_data())
    return position
