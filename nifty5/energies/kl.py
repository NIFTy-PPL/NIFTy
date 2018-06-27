from nifty5 import Energy, InversionEnabler, ScalingOperator, memo


class SampledKullbachLeiblerDivergence(Energy):
    def __init__(self, h, res_samples, iteration_controller):
        """
        h: Hamiltonian
        N: Number of samples to be used
        """
        super(SampledKullbachLeiblerDivergence, self).__init__(h.position)
        self._h = h
        self._res_samples = res_samples
        self._iteration_controller = iteration_controller

        self._energy_list = []
        for ss in res_samples:
            e = h.at(self.position+ss)
            self._energy_list.append(e)

    def at(self, position):
        return self.__class__(self._h.at(position), self._res_samples,
                              self._iteration_controller)


    @property
    @memo
    def value(self):
        v = self._energy_list[0].value
        for energy in self._energy_list[1:]:
            v += energy.value
        return v / len(self._energy_list)

    @property
    @memo
    def gradient(self):
        g = self._energy_list[0].gradient
        for energy in self._energy_list[1:]:
            g += energy.gradient
        return g / len(self._energy_list)

    @property
    @memo
    def curvature(self):
        # MR FIXME: This looks a bit strange...
        approx = self._energy_list[-1]._prior.curvature
        curvature_list = [e.curvature for e in self._energy_list]
        op = curvature_list[0]
        for curv in curvature_list[1:]:
            op = op + curv
        op = op * ScalingOperator(1./len(curvature_list), op.domain)
        return InversionEnabler(op, self._iteration_controller, approx)
