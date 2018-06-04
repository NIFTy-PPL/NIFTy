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


def generate_krylov_samples(D_inv, S, N_samps, controller):
    """
    Generates inverse samples from a curvature D.
    This algorithm iteratively generates samples from
    a curvature D by applying conjugate gradient steps
    and resampling the curvature in search direction.
    It is basically just a more stable version of
    Wiener Filter samples

    Parameters
    ----------
    D_inv : WienerFilterCurvature
        The curvature which will be the inverse of the covariance
        of the generated samples
    S : EndomorphicOperator (from which one can sample)
        A prior covariance operator which is used to generate prior
        samples that are then iteratively updated
    N_samps : Int
        How many samples to generate.
    controller : IterationController
        convergence controller for the conjugate gradient iteration

    Returns
    -------
    samples : a list of samples from D_inv.inverse
    """
    samples = []
    for i in range(N_samps):
        x0 = S.draw_sample()
        y = x0*0
        j = y*0
        #j = y
        energy = QuadraticEnergy(x0, D_inv, j)

        status = controller.start(energy)
        if status != controller.CONTINUE:
            samples += [y]
            break

        r = energy.gradient
        d = r.copy()

        previous_gamma = r.vdot(r).real
        if previous_gamma == 0:
            samples += [y+energy.position]
            break

        while True:
            q = energy.curvature(d)
            ddotq = d.vdot(q).real
            if ddotq == 0.:
                logger.error("Error: ConjugateGradient: ddotq==0.")
                samples += [y+energy.position]
                break
            alpha = previous_gamma/ddotq

            if alpha < 0:
                logger.error("Error: ConjugateGradient: alpha<0.")
                samples += [y+energy.position]
                break
    
            y += (np.random.randn()*np.sqrt(ddotq) )/ddotq * d

            q *= -alpha
            r = r + q

            energy = energy.at_with_grad(energy.position - alpha*d, r)

            gamma = r.vdot(r).real
            if gamma == 0:
                samples += [y+energy.position]
                break

            status = controller.check(energy)
            if status != controller.CONTINUE:
                samples += [y+energy.position]
                break

            d *= max(0, gamma/previous_gamma)
            d += r

            previous_gamma = gamma
    return samples
