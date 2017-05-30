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

import abc

from keepers import Loggable


class Transformation(Loggable, object):
    """
        A generic transformation which defines a static check_codomain
        method for all transforms.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, domain, codomain, module=None):
        if codomain is None:
            self.domain = domain
            self.codomain = self.get_codomain(domain)
        else:
            self.check_codomain(domain, codomain)
            self.domain = domain
            self.codomain = codomain

    @abc.abstractproperty
    def unitary(self):
        raise NotImplementedError

    @classmethod
    def get_codomain(cls, domain):
        raise NotImplementedError

    @classmethod
    def check_codomain(cls, domain, codomain):
        pass

    def transform(self, val, axes=None, **kwargs):
        raise NotImplementedError
