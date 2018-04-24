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

import numpy as np
from ..minimization.quadratic_energy import QuadraticEnergy


def generate_krylov_samples(D_inv, S, j, N_samps, controller):
    """
    Generates inverse samples from a curvature D.
    This algorithm iteratively generates samples from
    a curvature D by applying conjugate gradient steps
    and resampling the curvature in search direction.

    Parameters
    ----------
    D_inv : EndomorphicOperator
        The curvature which will be the inverse of the covarianc
        of the generated samples
    S : EndomorphicOperator (from which one can sample)
        A prior covariance operator which is used to generate prior
        samples that are then iteratively updated
    j : Field, optional
        A Field to which the inverse of D_inv is applied. The solution
        of this matrix inversion problem is a side product of generating
        the samples.
        If not supplied, it is sampled from the inverse prior.
    N_samps : Int
        How many samples to generate.
    controller : IterationController
        convergence controller for the conjugate gradient iteration

    Returns
    -------
    (solution, samples) : A tuple of a field 'solution' and a list of fields
        'samples'. The first entry of the tuple is the solution x to
            D_inv(x) = j
        and the second entry are a list of samples from D_inv.inverse
    """
    # MR FIXME: this should be synchronized with the "official" Nifty CG
    j = S.draw_sample(from_inverse=True) if j is None else j
    x = j*0.
    energy = QuadraticEnergy(x, D_inv, j)
    y = [S.draw_sample() for _ in range(N_samps)]

    status = controller.start(energy)
    if status != controller.CONTINUE:
        return x, y

    r = j.copy()
    p = r.copy()
    d = p.vdot(D_inv(p))
    while True:
        gamma = r.vdot(r)/d
        if gamma == 0.:
            break
        x = x + gamma*p
        Dip = D_inv(p)
        for samp in y:
            samp += (randn() * sqrt(d) - samp.vdot(Dip)) / d * p
        energy = energy.at(x)
        status = controller.check(energy)
        if status != controller.CONTINUE:
            return x, y
        r_new = r - gamma * Dip
        beta = r_new.vdot(r_new) / r.vdot(r)
        r = r_new
        p = r + beta * p
        d = p.vdot(Dip)
        if d == 0.:
            break
    return x, y
