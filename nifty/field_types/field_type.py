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

from nifty.domain_object import DomainObject


class FieldType(DomainObject):

    def weight(self, x, power=1, axes=None, inplace=False):
        if inplace:
            result = x
        else:
            result = x.copy()
        return result

    def process(self, method_name, array, inplace=True, **kwargs):
        try:
            result_array = self.__getattr__(method_name)(array,
                                                         inplace,
                                                         **kwargs)
        except AttributeError:
            if inplace:
                result_array = array
            else:
                result_array = array.copy()

        return result_array
