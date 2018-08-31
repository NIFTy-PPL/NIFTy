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

from __future__ import absolute_import, division, print_function

from ..compat import *
from ..operators.energy_operators import Hamiltonian, InverseGammaLikelihood
from ..operators.scaling_operator import ScalingOperator


def make_adjust_variances(a, xi, position, samples=[], scaling=None, ic_samp=None):
    """ Creates a Hamiltonian for constant likelihood optimizations.

    Constructs a Hamiltonian to solve constant likelihood optimizations of the
    form phi = a * xi under the constraint that phi remains constant.

    Parameters
    ----------
    a : Operator
        Operator which gives the amplitude when evaluated at a position
    xi : Operator
        Operator which gives the excitation when evaluated at a position
    postion : Field, MultiField
        Position of the whole problem
    samples : Field, MultiField
        Residual samples of the whole problem
    scaling : Float
        Optional rescaling of the Likelihood
    ic_samp : Controller
        Iteration Controller for Hamiltonian

    Returns
    -------
    Hamiltonian
        A Hamiltonian that can be used for further minimization
    """

    d = a*xi
    d = (d.conjugate()*d).real
    n = len(samples)
    if n > 0:
        d_eval = 0.
        for i in range(n):
            d_eval = d_eval + d(position + samples[i])
        d_eval = d_eval/n
    else:
        d_eval = d(position)

    x = (a.conjugate()*a).real
    if scaling is not None:
        x = ScalingOperator(scaling, x.target)(x)

    return Hamiltonian(InverseGammaLikelihood(x, d_eval), ic_samp=ic_samp)
