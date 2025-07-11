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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..minimization.energy_adapter import EnergyAdapter
from ..multi_field import MultiField
from ..operators.energy_operators import (InverseGammaEnergy,
                                          StandardHamiltonian)
from ..operators.scaling_operator import ScalingOperator
from ..operators.simple_linear_operators import ducktape


def make_adjust_variances_hamiltonian(a,
                                      xi,
                                      position,
                                      samples=[],
                                      scaling=None,
                                      ic_samp=None):
    """Creates a Hamiltonian for constant likelihood optimizations.

    Constructs a Hamiltonian to solve constant likelihood optimizations of the
    form phi = a * xi under the constraint that phi remains constant.

    xi is desired to be a Gaussian white Field, thus variations that are
    more easily represented by a should be absorbed in a.

    Parameters
    ----------
    a : Operator
        Gives the amplitude when evaluated at position.
    xi : Operator
        Field Adapter selecting a part of position.
        xi is desired to be a Gaussian white Field.
    position : :class:`nifty8.field.Field`, :class:`nifty8.multi_field.MultiField`
        Contains the initial values for the operators a and xi, to be adjusted
    samples : :class:`nifty8.field.Field`, :class:`nifty8.multi_field.MultiField`
        Residual samples of position.
    scaling : Float
        Optional rescaling of the Likelihood.
    ic_samp : Controller
        Iteration Controller for Hamiltonian.

    Returns
    -------
    :class:`nifty8.operators.energy_operators.StandardHamiltonian`
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
        x = ScalingOperator(x.target, scaling)(x)

    return StandardHamiltonian(InverseGammaEnergy(d_eval/2.)(x),
                               ic_samp=ic_samp)


def do_adjust_variances(position, A, minimizer, xi_key='xi', samples=[]):
    """Adjusts the variance of xi_key to be represented by amplitude_operator.

    Solves a constant likelihood optimization of the
    form phi = A * position[xi_key] under the constraint that phi remains
    constant.

    The field indexed by xi_key is desired to be a Gaussian white Field,
    thus variations that are more easily represented by A will be absorbed in
    A.

    Parameters
    ----------
    position : :class:`nifty8.field.Field`, :class:`nifty8.multi_field.MultiField`
        Contains the initial values for amplitude_operator and the key xi_key,
        to be adjusted.
    A : Operator
        Gives the amplitude when evaluated at position.
    minimizer : Minimizer
        Used to solve the optimization problem.
    xi_key : String
        Key of the Field containing undesired variations. This Field is
        contained in position.
    samples : :class:`nifty8.field.Field`, :class:`nifty8.multi_field.MultiField`, optional
        Residual samples of position. If samples are supplied then phi remains
        only approximately constant. Default: [].

    Returns
    -------
    MultiField
        The new position after variances have been adjusted.
    """
    xi = ducktape(None, position.domain, xi_key)
    ham = make_adjust_variances_hamiltonian(A, xi, position, samples=samples)

    # Minimize
    e = EnergyAdapter(
        position.extract(A.domain), ham, constants=[], want_metric=True)
    e, _ = minimizer(e)

    # Update position
    s_h_old = (A*xi).force(position)

    position = position.to_dict()
    position[xi_key] = s_h_old/A(e.position)
    position = MultiField.from_dict(position)
    return MultiField.union([position, e.position])
