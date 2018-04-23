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

from numpy import sqrt
from numpy.random import randn


def generate_krylov_samples(D_inv, S, j=None,  N_samps=1, N_iter=10,
                            name=None):
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
    N_samps : Int, optional
        How many samples to generate. Default: 1
    N_iter : Int, optional
        How many iterations of the conjugate gradient to run. Default: 10

    Returns
    -------
    (solution, samples) : A tuple of a field 'solution' and a list of fields
        'samples'. The first entry of the tuple is the solution x to
            D_inv(x) = j
        and the second entry are a list of samples from D_inv.inverse
    """
    j = S.draw_sample(from_inverse=True) if j is None else j
    x = j*0
    r = j.copy()
    p = r.copy()
    d = p.vdot(D_inv(p))
    y = [S.draw_sample() for _ in range(N_samps)]
    for k in range(1, 1+N_iter):
        gamma = r.vdot(r)/d
        if gamma == 0.:
            break
        x += gamma*p
        for i in range(N_samps):
            y[i] -= p.vdot(D_inv(y[i])) * p / d
            y[i] += randn() / sqrt(d) * p
        r_new = r - gamma * D_inv(p)
        beta = r_new.vdot(r_new) / r.vdot(r)
        r = r_new
        p = r + beta * p
        d = p.vdot(D_inv(p))
        if d == 0.:
            break
        if name is not None:
            print('{}: Iteration #{}'.format(name, k))
    return x, y
