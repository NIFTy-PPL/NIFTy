
import numpy as np
from nifty import Field,\
                  EndomorphicOperator,\
                  PowerSpace
class LaplaceOperator(EndomorphicOperator):
    def __init__(self, domain,
                 default_spaces=None):
        super(LaplaceOperator, self).__init__(default_spaces)
        if (domain is not None):
            if (not isinstance(domain, PowerSpace)):
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
        ret = 2 * array
        ret -= np.roll(array, 1)
        ret -= np.roll(array, -1)
        ret[0:2] = 0
        ret[-1] = 0
        return Field(self.domain, val=ret)

    def _adjoint_times(self, t, spaces):
        t = t.copy().weight(1)
        t.val[1:] /= self.domain[0].kindex[1:]
        if t.val.distribution_strategy != 'not':
            # self.logger.warn("distribution_strategy should be 'not' but is %s"
            #                  % t.val.distribution_strategy)
            array = t.val.get_full_data()
        else:
            array = t.val.get_local_data(copy=False)
        ret = np.copy(array)
        ret[0:2] = 0
        ret[-1] = 0
        ret = 2 * ret - np.roll(ret, 1) - np.roll(ret, -1)
        result = Field(self.domain, val=ret).weight(-1)
        return result



def _irregular_nabla(x,k):
    #applies forward differences and does nonesense at the edge. Thus needs cutting
    y = -x
    y[:-1] += x[1:]
    y[1:-1] /= - k[1:-1] + k[2:]
    return y

def _irregular_adj_nabla(z, k):
    #applies backwards differences*(-1) and does nonesense at the edge. Thus needs cutting
    x = z.copy()
    x[1:-1] /= - k[1:-1] + k[2:]
    y = -x
    y[1:] += x[:-1]
    return y


class LogLaplaceOperator(EndomorphicOperator):
    def __init__(self, domain,
                 default_spaces=None):
        super(LogLaplaceOperator, self).__init__(default_spaces)
        if (domain is not None):
            if (not isinstance(domain, PowerSpace)):
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
        l_vec = self.domain[0].kindex.copy()
        l_vec[1:] = np.log(l_vec[1:])
        l_vec[0] = -1.
        ret = t.val.copy()
        # ret[2:] *= np.sqrt(np.log(k_vec)-np.log(np.roll(k_vec,1)))[2:]
        ret = _irregular_nabla(ret,l_vec)
        ret = _irregular_adj_nabla(ret,l_vec)
        ret[0:2] = 0
        ret[-1] = 0
        ret[2:] *= np.sqrt(l_vec-np.roll(l_vec,1))[2:]
        return Field(self.domain, val=ret).weight(power=-0.5)

    def _adjoint_times(self, t, spaces):
        ret = t.copy().weight(power=0.5).val
        l_vec = self.domain[0].kindex.copy()
        l_vec[1:] = np.log(l_vec[1:])
        l_vec[0] = -1.
        ret[2:] *= np.sqrt(l_vec-np.roll(l_vec,1))[2:]
        ret[0:2] = 0
        ret[-1] = 0
        ret = _irregular_nabla(ret,l_vec)
        ret = _irregular_adj_nabla(ret,l_vec)
        # ret[2:] *= np.sqrt(np.log(k_vec)-np.log(np.roll(k_vec,1)))[2:]
        return Field(self.domain, val=ret).weight(-1)