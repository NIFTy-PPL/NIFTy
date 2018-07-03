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

from ..library.gaussian_energy import GaussianEnergy
from ..minimization.energy import Energy
from ..models.variable import Variable
from ..operators.sampling_enabler import SamplingEnabler
from ..utilities import memo


class Hamiltonian(Energy):
    def __init__(self, lh, iteration_controller_sampling=None):
        """
        lh: Likelihood (energy object)
        prior:
        """
        super(Hamiltonian, self).__init__(lh.position)
        self._lh = lh
        self._ic_samp = iteration_controller_sampling
        self._prior = GaussianEnergy(Variable(self.position))

    def at(self, position):
        return self.__class__(self._lh.at(position), self._ic_samp)

    @property
    @memo
    def value(self):
        return self._lh.value + self._prior.value

    @property
    @memo
    def gradient(self):
        return self._lh.gradient + self._prior.gradient

    @property
    @memo
    def curvature(self):
        prior_curv = self._prior.curvature
        if self._ic_samp is None:
            return self._lh.curvature + prior_curv
        else:
            return SamplingEnabler(self._lh.curvature, prior_curv.inverse,
                                   self._ic_samp, prior_curv.inverse)

    def __str__(self):
        res = 'Likelihood:\t{:.2E}\n'.format(self._lh.value)
        res += 'Prior:\t\t{:.2E}'.format(self._prior.value)
        return res
