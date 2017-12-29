from .endomorphic_operator import EndomorphicOperator
from .laplace_operator import LaplaceOperator


class SmoothnessOperator(EndomorphicOperator):
    """An operator measuring the smoothness on an irregular grid with respect
    to some scale.

    This operator applies the irregular LaplaceOperator and its adjoint to some
    Field over a PowerSpace which corresponds to its smoothness and weights the
    result with a scale parameter sigma. It is used in the smoothness prior
    terms of the CriticalPowerEnergy. For this purpose we use free boundary
    conditions in the LaplaceOperator, having no curvature at both ends. In
    addition the first entry is ignored as well, corresponding to the overall
    mean of the map. The mean is therefore not considered in the smoothness
    prior.


    Parameters
    ----------
    strength: float,
        Specifies the strength of the SmoothnessOperator
    logarithmic : boolean,
        Whether smoothness is calculated on a logarithmic scale or linear scale
        default : True
    """

    def __init__(self, domain, strength=1., logarithmic=True, space=None):
        super(SmoothnessOperator, self).__init__()
        self._laplace = LaplaceOperator(domain, logarithmic=logarithmic,
                                        space=space)

        if strength < 0:
            raise ValueError("ERROR: strength must be >=0.")
        self._strength = strength

    @property
    def domain(self):
        return self._laplace._domain

    # MR FIXME: shouldn't this operator actually be self-adjoint?
    @property
    def capability(self):
        return self.TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)

        if self._strength == 0.:
            return x.zeros_like(x)
        result = self._laplace.adjoint_times(self._laplace(x))
        result *= self._strength**2
        return result

    @property
    def logarithmic(self):
        return self._laplace.logarithmic

    @property
    def strength(self):
        return self._strength
