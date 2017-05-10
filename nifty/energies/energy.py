# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

from nifty.nifty_meta import NiftyMeta

from keepers import Loggable


class Energy(Loggable, object):
    __metaclass__ = NiftyMeta

    def __init__(self, position):
        self._cache = {}
        try:
            position = position.copy()
        except AttributeError:
            pass
        self.position = position

    def at(self, position):
        return self.__class__(position)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position

    @property
    def value(self):
        raise NotImplementedError

    @property
    def gradient(self):
        raise NotImplementedError

    @property
    def curvature(self):
        raise NotImplementedError
