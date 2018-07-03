from builtins import *
from ..minimization.energy import Energy
from ..utilities import memo, my_sum


class SampledKullbachLeiblerDivergence(Energy):
    def __init__(self, h, res_samples):
        """
        h: Hamiltonian
        N: Number of samples to be used
        """
        super(SampledKullbachLeiblerDivergence, self).__init__(h.position)
        self._h = h
        self._res_samples = res_samples

        self._energy_list = tuple(h.at(self.position+ss)
                                  for ss in res_samples)

    def at(self, position):
        return self.__class__(self._h.at(position), self._res_samples)

    @property
    @memo
    def value(self):
        return (my_sum(map(lambda v: v.value, self._energy_list)) /
                len(self._energy_list))

    @property
    @memo
    def gradient(self):
        return (my_sum(map(lambda v: v.gradient, self._energy_list)) /
                len(self._energy_list))

    @property
    @memo
    def curvature(self):
        return (my_sum(map(lambda v: v.curvature, self._energy_list)) *
                (1./len(self._energy_list)))
