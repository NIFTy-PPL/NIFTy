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

from functools import reduce

from .domain import Domain


class UnstructuredDomain(Domain):
    """A :class:`~nifty8.domains.domain.Domain` subclass for spaces with no
    associated geometry.

    Typically used for data spaces.

    Parameters
    ----------
    shape : tuple of int
        The required shape for an array which can hold the unstructured
        domain's data.
    """

    _needed_for_hash = ["_shape"]

    def __init__(self, shape):
        try:
            self._shape = tuple([int(i) for i in shape])
        except TypeError:
            self._shape = (int(shape), )

    def __repr__(self):
        return "UnstructuredDomain(shape={})".format(self.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return reduce(lambda x, y: x*y, self.shape)
