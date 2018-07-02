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
from .model import Model


class Constant(Model):
    """A sky model with a constant (multi-)field as value.

    Parameters
    ----------
    position : Field or MultiField
        The current position in parameter space.
    constant : Field
        The value of the model.

    Note
    ----
    Since there is no model-function associated:
        - Position has no influence on value.
        - There is no gradient.
    """
    # TODO Remove position
    def __init__(self, position, constant):
        super(Constant, self).__init__(position)
        self._constant = constant

        self._value = self._constant
        self._gradient = 0.

    def at(self, position):
        return self.__class__(position, self._constant)
