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
from ..field import Field

__all__ = ["check_value_gradient_consistency"]


def check_value_gradient_consistency(E, tol=1e-6, ntries=100):
    if not np.isfinite(E.value):
        raise ValueError
    for _ in range(ntries):
        dir = Field.from_random("normal", E.position.domain)
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
        Enext = E2
        dirder = E.gradient.vdot(dir)
        for i in range(50):
            Ediff = E2.value - E.value
            eps = 1e-10*max(abs(E.value), abs(E2.value))
            if abs(Ediff-dirder) < max([tol*abs(Ediff), tol*abs(dirder), eps]):
                break
            dir *= 0.5
            dirder *= 0.5
            E2 = E2.at(E.position+dir)
        else:
            raise ValueError("gradient and value seem inconsistent")
        E = Enext
