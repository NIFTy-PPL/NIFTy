
import numpy as np
from nifty import Field,\
                  EndomorphicOperator,\
                  PowerSpace

# def _irregular_nabla(x,k):
#     #applies forward differences and does nonesense at the edge. Thus needs cutting
#     y = -x
#     y[:-1] += x[1:]
#     y[1:-1] /= - k[1:-1] + k[2:]
#     return y
#
# def _irregular_adj_nabla(z, k):
#     #applies backwards differences*(-1) and does nonesense at the edge. Thus needs cutting
#     x = z.copy()
#     x[1:-1] /= - k[1:-1] + k[2:]
#     y = -x
#     y[1:] += x[:-1]
#     return y


class LaplaceOperator(EndomorphicOperator):
    """A irregular LaplaceOperator with free boundary and excluding monopole.

    This LaplaceOperator implements the second derivative of a Field in PowerSpace
    on logarithmic or linear scale with vanishing curvature at the boundary, starting
    at the second entry of the Field. The second derivative of the Field on the irregular grid
    is calculated using finite differences.

    Parameters
    ----------
    logarithmic : boolean,
        Whether smoothness is calculated on a logarithmic scale or linear scale
        default : True
    """

    def __init__(self, domain,
                 default_spaces=None, logarithmic = True):
        super(LaplaceOperator, self).__init__(default_spaces)
        if (domain is not None):
            if (not isinstance(domain, PowerSpace)):
                raise TypeError("The domain has to live over a PowerSpace")
        self._domain = self._parse_domain(domain)
        if logarithmic :
            self.positions = self.domain[0].kindex.copy()
            self.positions[1:] = np.log(self.positions[1:])
            self.positions[0] = -1.
        else :
            self.positions = self.domain[0].kindex.copy()
            self.positions[0] = -1

        self.fwd_dist = self.positions[1:] - self.positions[:-1]

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
        ret = t.val.copy()
        ret = self._irregular_laplace(ret)
        ret[0:2] = 0
        ret[-1] = 0
        ret[2:] *= np.sqrt(self.fwd_dist)[1:]
        return Field(self.domain, val=ret).weight(power=-0.5)

    def _adjoint_times(self, t, spaces):
        ret = t.copy().weight(power=0.5).val
        ret[2:] *= np.sqrt(self.fwd_dist)[1:]
        ret[0:2] = 0
        ret[-1] = 0
        ret = self._irregular_adj_laplace(ret)
        return Field(self.domain, val=ret).weight(-1)

    def _irregular_laplace(self, x):
        ret = np.zeros_like(x)
        ret[1:-1] = -(x[1:-1] - x[0:-2]) / self.fwd_dist[:-1] \
                    + (x[2:] - x[1:-1]) / self.fwd_dist[1:]
        ret[1:-1] /= self.positions[2:] - self.positions[:-2]
        ret *= 2.
        return ret

    def _irregular_adj_laplace(self, x):
        ret = np.zeros_like(x)
        y = x.copy()
        y[1:-1] /= self.positions[2:] - self.positions[:-2]
        y *= 2
        ret[1:-1] = -y[1:-1] / self.fwd_dist[:-1] - y[1:-1] / self.fwd_dist[1:]
        ret[0:-2] += y[1:-1] / self.fwd_dist[:-1]
        ret[2:] += y[1:-1] / self.fwd_dist[1:]
        return ret
