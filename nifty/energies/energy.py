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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from ..nifty_meta import NiftyMeta
from .memoization import memo

from future.utils import with_metaclass


class Energy(with_metaclass(NiftyMeta, type('NewBase', (object,), {}))):
    """ Provides the functional used by minimization schemes.

   The Energy object is an implementation of a scalar function including its
   gradient and curvature at some position.

    Parameters
    ----------
    position : Field
        The input parameter of the scalar function.

    Attributes
    ----------
    position : Field
        The Field location in parameter space where value, gradient and
        curvature are evaluated.
    value : np.float
        The value of the energy functional at given `position`.
    gradient : Field
        The gradient at given `position`.
    curvature : LinearOperator, callable
        A positive semi-definite operator or function describing the curvature
        of the potential at the given `position`.

    Notes
    -----
    An instance of the Energy class is defined at a certain location. If one
    is interested in the value, gradient or curvature of the abstract energy
    functional one has to 'jump' to the new position using the `at` method.
    This method returns a new energy instance residing at the new position. By
    this approach, intermediate results from computing e.g. the gradient can
    safely be reused for e.g. the value or the curvature.

    Memorizing the evaluations of some quantities (using the memo decorator)
    minimizes the computational effort for multiple calls.

    See Also
    --------
    memo

    """

    def __init__(self, position):
        super(Energy, self).__init__()
        self._position = position.copy()

    def at(self, position):
        """ Initializes and returns a new Energy object at the new position.

        Parameters
        ----------
        position : Field
            Parameter for the new Energy object.

        Returns
        -------
        out : Energy
            Energy object at new position.

        """

        return self.__class__(position)

    @property
    def position(self):
        """
        The Field location in parameter space where value, gradient and
        curvature are evaluated.

        """

        return self._position

    @property
    def value(self):
        """
        The value of the energy functional at given `position`.

        """

        raise NotImplementedError

    @property
    def gradient(self):
        """
        The gradient at given `position`.

        """

        raise NotImplementedError

    @property
    @memo
    def gradient_norm(self):
        """
        The length of the gradient at given `position`.

        """

        return self.gradient.norm()

    @property
    @memo
    def gradient_infnorm(self):
        """
        The infinity norm of the gradient at given `position`.

        """

        return abs(self.gradient).max()

    @property
    def curvature(self):
        """
        A positive semi-definite operator or function describing the curvature
        of the potential at the given `position`.

        """

        raise NotImplementedError
