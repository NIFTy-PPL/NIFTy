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


class _TransformationCache(object):
    def __init__(self):
        self.cache = {}

    def create(self, transformation_class, domain, codomain, module):
        key = domain.__hash__() ^ ((codomain.__hash__()/111) ^
                                   (module.__hash__())/179)
        if key not in self.cache:
            self.cache[key] = transformation_class(domain, codomain, module)

        return self.cache[key]

TransformationCache = _TransformationCache()
