
import numpy as np
from nifty import Field,\
                  EndomorphicOperator,\
                  PowerSpace
class LaplaceOperator(EndomorphicOperator):
    def __init__(self, domain,
                 default_spaces=None):
        super(LaplaceOperator, self).__init__(default_spaces)
        if (domain is not None):
            if(not isinstance(domain, PowerSpace)):
                    raise TypeError("The domain has to live over a PowerSpace")
        self._domain = self._parse_domain(domain)
    """
    input parameters:
    domain- can only live over one domain
    to do:
    correct implementation of D20 object
    """
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
        if t.val.distribution_strategy != 'not':
            # self.logger.warn("distribution_strategy should be 'not' but is %s"
            #                  % t.val.distribution_strategy)
            array = t.val.get_full_data()
        else:
            array = t.val.get_local_data(copy=False)
        ret = 2 * array - np.append(0, array[:-1]) - np.append(array[1:], 0)
        ret[0] = 0
        ret[1] = 0
        ret[-1] = 0
        return Field(self.domain, val=ret)
    def _adjoint_times(self, t, spaces):
        if t.val.distribution_strategy != 'not':
            # self.logger.warn("distribution_strategy should be 'not' but is %s"
            #                  % t.val.distribution_strategy)
            array = t.val.get_full_data()
        else:
            array = t.val.get_local_data(copy=False)
        ret = 2 * array - np.append(0, array[:-1]) - np.append(array[1:], 0)
        ret[0] = 0
        ret[1] = -array[2]
        ret[2] = 2*array[2]-array[3]
        ret[-1] = -array[-2]
        ret[-2] = -array[-3]+2*array[-2]
        return Field(self.domain, val=ret)