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

__all__ = ["check_value_gradient_consistency",
           "check_value_gradient_curvature_consistency"]


def _get_acceptable_energy(E):
    val = E.value
    if not np.isfinite(val):
        raise ValueError
    dir = from_random("normal", E.position.domain)
    dirder = E.gradient.vdot(dir)
    dir *= np.abs(val)/np.abs(dirder)*1e-5
    # find a step length that leads to a "reasonable" energy
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


def check_value_gradient_consistency(E, tol=1e-6, ntries=100):
    for _ in range(ntries):
        E2 = _get_acceptable_energy(E)
        val = E.value
        dir = E2.position - E.position
        # Enext = E2
        dirnorm = dir.norm()
        for i in range(50):
            Emid = E.at(E.position + 0.5*dir)
            dirder = Emid.gradient.vdot(dir)/dirnorm
            if abs((E2.value-val)/dirnorm-dirder) < tol:
                break
            dir *= 0.5
            dirnorm *= 0.5
            E2 = Emid
        else:
            raise ValueError("gradient and value seem inconsistent")
        # E = Enext


def check_value_gradient_curvature_consistency(E, tol=1e-6, ntries=100):
    for _ in range(ntries):
        E2 = _get_acceptable_energy(E)
        val = E.value
        dir = E2.position - E.position
        # Enext = E2
        dirnorm = dir.norm()
        for i in range(50):
            Emid = E.at(E.position + 0.5*dir)
            dirder = Emid.gradient.vdot(dir)/dirnorm
            dgrad = Emid.curvature(dir)/dirnorm
            if abs((E2.value-val)/dirnorm-dirder) < tol and \
               (abs((E2.gradient-E.gradient)/dirnorm-dgrad) < tol).all():
                break
            dir *= 0.5
            dirnorm *= 0.5
            E2 = Emid
        else:
            raise ValueError("gradient, value and curvature seem inconsistent")
        # E = Enext
