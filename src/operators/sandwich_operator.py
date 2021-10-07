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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from .endomorphic_operator import EndomorphicOperator
from .linear_operator import LinearOperator
from .scaling_operator import ScalingOperator


class SandwichOperator(EndomorphicOperator):
    """Operator which is equivalent to the expression
    `bun.adjoint(cheese(bun))`.

    Note
    ----
    This operator should always be called using the `make` method.
    """

    def __init__(self, bun, cheese, op, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        self._bun = bun
        self._cheese = cheese
        self._op = op
        self._domain = op.domain
        self._capability = op._capability

    @staticmethod
    def make(bun, cheese=None, sampling_dtype=None):
        """Build a SandwichOperator (or something simpler if possible)

        Parameters
        ----------
        bun: LinearOperator
            the bun part
        cheese: EndomorphicOperator
            the cheese part
        sampling_dtype :
            If this operator represents the covariance of a Gaussian probabilty
            distribution and cheese is `None`, `sampling_dtype` specifies if it
            is real or complex Gaussian. If `sampling_dtype` and `cheese` are
            `None`, the operator cannot be used as a covariance, i.e. no samples
            can be drawn. Default: None.
        """
        if isinstance(cheese, SandwichOperator):
            old_cheese = cheese
            cheese = old_cheese._cheese
            bun = old_cheese._bun @ bun

        if not isinstance(bun, LinearOperator):
            raise TypeError("bun must be a linear operator")
        if cheese is not None and not isinstance(cheese, LinearOperator):
            raise TypeError("cheese must be a linear operator or None")
        if cheese is None:
            cheese = ScalingOperator(bun.target, 1., sampling_dtype)
        if isinstance(bun, ScalingOperator):
            fct = abs(bun._factor)**2
            if fct == 1.:
                return cheese
            op = cheese.scale(fct)
        else:
            op = bun.adjoint @ cheese @ bun
        return SandwichOperator(bun, cheese, op, _callingfrommake=True)

    def apply(self, x, mode):
        return self._op.apply(x, mode)

    def draw_sample(self, from_inverse=False):
        # Inverse samples from general sandwiches are not possible
        if from_inverse:
            if self._bun.capability & self._bun.INVERSE_TIMES:
                try:
                    s = self._cheese.draw_sample(from_inverse)
                    return self._bun.inverse_times(s)
                except NotImplementedError:
                    pass
            raise NotImplementedError(
                "cannot draw from inverse of this operator")

        # Samples from general sandwiches
        return self._bun.adjoint_times(
            self._cheese.draw_sample(from_inverse))

    def get_sqrt(self):
        if self._cheese is None:
            return self._bun
        return self._cheese.get_sqrt() @ self._bun

    def __repr__(self):
        from ..utilities import indent
        return "\n".join((
            "SandwichOperator:",
            indent("\n".join((
                "Cheese:", self._cheese.__repr__(),
                "Bun:", self._bun.__repr__())))))
