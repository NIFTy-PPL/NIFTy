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
from ..sugar import from_random
from .. import Energy, Model

__all__ = ["check_value_gradient_consistency",
           "check_value_gradient_curvature_consistency"]


def _get_acceptable_model(M):
    val = M.value
    if not np.isfinite(val.sum()):
        raise ValueError('Initial Model value must be finite')
    dir = from_random("normal", M.position.domain)
    dirder = M.gradient(dir)
    dir *= val/(dirder).norm()*1e-5
    # Find a step length that leads to a "reasonable" Model
    for i in range(50):
        try:
            M2 = M.at(M.position+dir)
            if np.isfinite(M2.value.sum()) and abs(M2.value.sum()) < 1e20:
                break
        except FloatingPointError:
            pass
        dir *= 0.5
    else:
        raise ValueError("could not find a reasonable initial step")
    return M2


def _get_acceptable_energy(E):
    val = E.value
    if not np.isfinite(val):
        raise ValueError('Initial Energy must be finite')
    dir = from_random("normal", E.position.domain)
    dirder = E.gradient.vdot(dir)
    dir *= np.abs(val)/np.abs(dirder)*1e-5
    # Find a step length that leads to a "reasonable" energy
    for i in range(50):
        try:
            E2 = E.at(E.position+dir)
            if np.isfinite(E2.value) and abs(E2.value) < 1e20:
                break
        except FloatingPointError:
            pass
        dir *= 0.5
    else:
        raise ValueError("could not find a reasonable initial step")
    return E2


def check_value_gradient_consistency(E, tol=1e-8, ntries=100):
    for _ in range(ntries):
        if isinstance(E, Energy):
            E2 = _get_acceptable_energy(E)
        else:
            E2 = _get_acceptable_model(E)
        val = E.value
        dir = E2.position - E.position
        dirnorm = dir.norm()
        for i in range(50):
            Emid = E.at(E.position + 0.5*dir)
            if isinstance(E, Energy):
                dirder = Emid.gradient.vdot(dir)/dirnorm
            else:
                dirder = Emid.gradient(dir)/dirnorm
            numgrad = (E2.value-val)/dirnorm
            if isinstance(E, Model):
                xtol = tol*dirder.norm()
                if (abs(numgrad-dirder) < xtol).all():
                    break
            else:
                xtol = tol*Emid.gradient_norm
                if abs(numgrad-dirder) < xtol:
                    break
            dir *= 0.5
            dirnorm *= 0.5
            E2 = Emid
        else:
            raise ValueError("gradient and value seem inconsistent")


def check_value_gradient_curvature_consistency(E, tol=1e-8, ntries=100):
    if isinstance(E, Model):
        raise ValueError('Models have no curvature, thus it cannot be tested.')
    for _ in range(ntries):
        E2 = _get_acceptable_energy(E)
        val = E.value
        dir = E2.position - E.position
        dirnorm = dir.norm()
        for i in range(50):
            Emid = E.at(E.position + 0.5*dir)
            dirder = Emid.gradient.vdot(dir)/dirnorm
            dgrad = Emid.curvature(dir)/dirnorm
            xtol = tol*Emid.gradient_norm
            if abs((E2.value-val)/dirnorm - dirder) < xtol and \
               (abs((E2.gradient-E.gradient)/dirnorm-dgrad) < xtol).all():
                break
            dir *= 0.5
            dirnorm *= 0.5
            E2 = Emid
        else:
            raise ValueError("gradient, value and curvature seem inconsistent")
