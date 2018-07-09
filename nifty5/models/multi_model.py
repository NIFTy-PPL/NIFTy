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

from __future__ import absolute_import, division, print_function
from ..compat import *
from ..domain_tuple import DomainTuple
from ..multi.multi_field import MultiField
from ..operators.multi_adaptor import MultiAdaptor
from .model import Model


class MultiModel(Model):
    """ """
    def __init__(self, model, key):
        # TODO Rewrite it such that it takes a dictionary as input.
        # (just like MultiFields).
        super(MultiModel, self).__init__(model.position)
        self._model = model
        self._key = key
        val = self._model.value
        if not isinstance(val.domain, DomainTuple):
            raise TypeError
        self._value = MultiField({key: val})
        self._jacobian = (MultiAdaptor(self.value.domain) *
                          self._model.jacobian)

    def at(self, position):
        return self.__class__(self._model.at(position), self._key)
