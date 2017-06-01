from nifty import EndomorphicOperator,\
                  PowerSpace

from laplace_operator import LogLaplaceOperator


class SmoothnessOperator(EndomorphicOperator):

    def __init__(self, domain, sigma=1.,
                 default_spaces=None):

        super(SmoothnessOperator, self).__init__(default_spaces)

        if (not isinstance(domain, PowerSpace)):
            raise TypeError("The domain has to live over a PowerSpace")

        self._domain = self._parse_domain(domain)

        if (sigma <= 0):
            raise ValueError("ERROR: invalid sigma.")

        self._sigma = sigma

        self._Laplace = LogLaplaceOperator(domain=self._domain[0])

    """
    SmoothnessOperator

    input parameters:
    domain: Domain of the field, has to be a single PowerSpace
    sigma: specifying the expected roughness, has to be a positive float

    """

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
