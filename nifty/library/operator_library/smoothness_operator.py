from nifty import EndomorphicOperator,\
                  PowerSpace

from laplace_operator import LaplaceOperator


class SmoothnessOperator(EndomorphicOperator):
    """An operator measuring the smoothness on an irregular grid with respect to some scale.

    This operator applies the irregular LaplaceOperator and its adjoint to some Field over a
    PowerSpace which corresponds to its smoothness and weights the result with a scale parameter sigma.
    It is used in the smoothness prior terms of the CriticalPowerEnergy. For this purpose we
    use free boundary conditions in the LaplaceOperator, having no curvature at both ends.
    In addition the first entry is ignored as well, corresponding to the overall mean of the map.
    The mean is therefore not considered in the smoothness prior.


    Parameters
    ----------
    sigma: float,
        Specifies the strength of the SmoothnessOperator
    logarithmic : boolean,
        Whether smoothness is calculated on a logarithmic scale or linear scale
        default : True
    """

    def __init__(self, domain, sigma ,logarithmic = True,
                 default_spaces=None):

        super(SmoothnessOperator, self).__init__(default_spaces)

        if (not isinstance(domain, PowerSpace)):
            raise TypeError("The domain has to live over a PowerSpace")

        self._domain = self._parse_domain(domain)

        if (sigma <= 0):
            raise ValueError("ERROR: invalid sigma.")

        self._sigma = sigma

        self._Laplace = LaplaceOperator(domain=self._domain[0], logarithmic=logarithmic)



    @property
    def sigma(self):
        return self._sigma

    @property
    def target(self):
        return self._domain

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

    def _times(self, t, spaces):
        res = self._Laplace.adjoint_times(self._Laplace.times(t))
        return (1/self.sigma)**2*res
