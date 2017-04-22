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

import abc

import numpy as np

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

    @classmethod
    def get_codomain(cls, domain, dtype=None, zerocenter=None):
        raise NotImplementedError

    @classmethod
    def check_codomain(cls, domain, codomain):
        if np.dtype(domain.dtype) != np.dtype(codomain.dtype):
            cls.Logger.warn("Unrecommended: domain and codomain don't have "
                            "the same dtype.")

    def transform(self, val, axes=None, **kwargs):
        raise NotImplementedError
