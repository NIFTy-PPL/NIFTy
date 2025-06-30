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

import numpy as np

from .structured_domain import StructuredDomain


class DOFSpace(StructuredDomain):
    """Generic degree-of-freedom space. It is defined as the domain of some
    DOFDistributor.
    Its entries represent the underlying degrees of freedom of some other
    space, according to the dofdex.

    Parameters
    ----------
    dof_weights: 1-D numpy array
        A numpy array containing the multiplicity of each individual degree of
        freedom.
    """

    _needed_for_hash = ["_dvol"]

    def __init__(self, dof_weights):
        self._dvol = tuple(dof_weights)

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (len(self._dvol),)

    @property
    def size(self):
        return len(self._dvol)

    @property
    def scalar_dvol(self):
        return None

    @property
    def dvol(self):
        return np.array(self._dvol)

    def __repr__(self):
        return f'DOFSpace(dof_weights={self._dvol})'
