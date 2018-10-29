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

import numpy as np

from .. import dobj
from ..compat import *
from ..field import Field
from ..sugar import exp
from .structured_domain import StructuredDomain


class LogRGSpace(StructuredDomain):
    """NIFTy subclass for logarithmic Cartesian grids.

    Parameters
    ----------
    shape : int or tuple of int
        Number of grid points or numbers of gridpoints along each axis.
    bindistances : float or tuple of float
        Distance between two grid points along each axis. These are
        measured on logarithmic scale and are constant therfore.
    t_0 : float or tuple of float
        FIXME
    harmonic : bool, optional
        Whether the space represents a grid in position or harmonic space.
        (default: False).
    """
    _needed_for_hash = ['_shape', '_bindistances', '_t_0', '_harmonic']

    def __init__(self, shape, bindistances, t_0, harmonic=False):
        self._harmonic = bool(harmonic)

        if np.isscalar(shape):
            shape = (shape,)
        self._shape = tuple(int(i) for i in shape)

        self._bindistances = tuple(bindistances)
        self._t_0 = tuple(t_0)

        self._dim = int(reduce(lambda x, y: x*y, self._shape))
        self._dvol = float(reduce(lambda x, y: x*y, self._bindistances))

    @property
    def harmonic(self):
        return self._harmonic

    @property
    def shape(self):
        return self._shape

    @property
    def scalar_dvol(self):
        return self._dvol

    @property
    def bindistances(self):
        return np.array(self._bindistances)

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def t_0(self):
        return np.array(self._t_0)

    def __repr__(self):
        return ("LogRGSpace(shape={}, harmonic={})".format(
            self.shape, self.harmonic))

    def get_default_codomain(self):
        codomain_bindistances = 1./(self.bindistances*self.shape)
        return LogRGSpace(self.shape, codomain_bindistances, self._t_0, True)

    def get_k_length_array(self):
        if not self.harmonic:
            raise NotImplementedError
        ks = self.get_k_array()
        return Field.from_global_data(self, np.linalg.norm(ks, axis=0))

    def get_k_array(self):
        ndim = len(self.shape)
        k_array = np.zeros((ndim,) + self.shape)
        dist = self.bindistances
        for i in range(ndim):
            ks = np.zeros(self.shape[i])
            ks[1:] = np.minimum(self.shape[i] - 1 - np.arange(self.shape[i]-1), np.arange(self.shape[i]-1)) * dist[i]
            if self.harmonic:
                ks[0] = np.nan
            else:
                ks[0] = -np.inf
                ks[1:] += self.t_0[i]
            k_array[i] += ks.reshape((1,)*i + (self.shape[i],) + (1,)*(ndim-i-1))
        return k_array
