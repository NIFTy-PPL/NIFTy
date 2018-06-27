import numpy as np
from scipy.stats import invgamma, norm
from ..field import Field
from ..sugar import makeOp
from ..multi import MultiField
from ..model import Model

from ..operators import SelectionOperator
from ..utilities import memo


class PointSources(Model):
    def __init__(self, position, alpha, q):
        super(PointSources, self).__init__(position)
        self._alpha = alpha
        self._q = q

    def at(self, position):
        return self.__class__(position, self._alpha, self._q)

    @property
    @memo
    def value(self):
        points = self.position['points'].to_global_data()
        points = np.clip(points, None, 8.2)
        points = Field(self.position['points'].domain, points)
        return self.IG(points, self._alpha, self._q)

    @property
    @memo
    def gradient(self):
        u = self.position['points']
        inner = norm.pdf(u.val)
        outer_inv = invgamma.pdf(invgamma.ppf(norm.cdf(u.val),
                                              self._alpha,
                                              scale=self._q),
                                 self._alpha, scale=self._q)
        # FIXME
        outer_inv = np.clip(outer_inv, 1e-20, None)
        outer = 1/outer_inv
        grad = Field(u.domain, val=inner*outer)
        grad = makeOp(MultiField({'points': grad}))
        return SelectionOperator(grad.target, 'points')*grad

    @staticmethod
    def IG(field, alpha, q):
        foo = invgamma.ppf(norm.cdf(field.val), alpha, scale=q)
        return Field(field.domain, val=foo)

    @staticmethod
    def IG_prime(field, alpha, q):
        inner = norm.pdf(field.val)
        outer = invgamma.pdf(invgamma.ppf(norm.cdf(field.val), alpha, scale=q), alpha, scale=q)
        # # FIXME
        # outer = np.clip(outer, 1e-20, None)
        outer = 1/outer
        return Field(field.domain, val=inner*outer)

    @staticmethod
    def inverseIG(u, alpha, q):
        res = norm.ppf(invgamma.cdf(u, alpha, scale=q))
        # # FIXME
        # res = np.clip(res, 0, None)
        return res
