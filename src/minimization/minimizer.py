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

from ..utilities import NiftyMeta


class Minimizer(metaclass=NiftyMeta):
    """A base class used by all minimizers."""

    def __call__(self, energy, preconditioner=None):
        """Performs the minimization of the provided Energy functional.

        Parameters
        ----------
        energy : Energy
           Energy object at the starting point of the iteration

        preconditioner : LinearOperator, optional
           Preconditioner to accelerate the minimization

        Returns
        -------
        Energy : Latest `energy` of the minimization.
        int : exit status of the minimization
            Can be controller.CONVERGED or controller.ERROR
        """
        raise NotImplementedError

    @property
    def controller(self):
        raise NotImplementedError
