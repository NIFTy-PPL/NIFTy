from ...spaces.power_space import PowerSpace
from ..endomorphic_operator import EndomorphicOperator
from ..laplace_operator import LaplaceOperator


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

    # ---Overwritten properties and methods---

    def __init__(self, domain, strength=1., logarithmic=True,
                 default_spaces=None):

        super(SmoothnessOperator, self).__init__(default_spaces=default_spaces)

        self._domain = self._parse_domain(domain)
        if len(self.domain) != 1:
            raise ValueError("The domain must contain exactly one PowerSpace.")

        if not isinstance(self.domain[0], PowerSpace):
            raise TypeError("The domain must contain exactly one PowerSpace.")

        if strength < 0:
            raise ValueError("ERROR: invalid sigma.")

        self._strength = strength

        self._laplace = LaplaceOperator(domain=self.domain,
                                        logarithmic=logarithmic)

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._domain

    @property
    def unitary(self):
        return False

    @property
    def symmetric(self):
        return False

    @property
    def self_adjoint(self):
        return False

    def _times(self, x, spaces):
        if self._strength != 0:
            result = self._laplace.adjoint_times(self._laplace(x, spaces),
                                                 spaces)
            result *= self._strength**2
        else:
            result = x.copy_empty()
            result.val[:] = 0
        return result

    # ---Added properties and methods---

    @property
    def logarithmic(self):
        return self._laplace.logarithmic

    @property
    def strength(self):
        return self._strength
