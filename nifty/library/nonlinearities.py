from numpy import logical_and, where
from ... import Field, exp, tanh


class Linear:
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1


class Exponential(object):
    def __call__(self, x):
        return exp(x)

    def derivative(self, x):
        return exp(x)


class Tanh(object):
    def __call__(self, x):
        return tanh(x)

    def derivative(self, x):
        return (1. - tanh(x)**2)


class PositiveTanh:
    def __call__(self, x):
        return 0.5 * tanh(x) + 0.5

    def derivative(self, x):
        return 0.5 * (1. - tanh(x)**2)


class LinearThenQuadraticWithJump(object):
    def __call__(self, x):
        dom = x.domain
        x = x.copy().val.get_full_data()
        cond = where(x > 0.)
        not_cond = where(x <= 0.)
        x[cond] **= 2
        x[not_cond] -= 1
        return Field(domain=dom, val=x)

    def derivative(self, x):
        dom = x.domain
        x = x.copy().val.get_full_data()
        cond = where(x > 0.)
        not_cond = where(x <= 0.)
        x[cond] *= 2
        x[not_cond] = 1
        return Field(domain=dom, val=x)


class ReallyStupidNonlinearity(object):
    def __call__(self, x):
        dom = x.domain
        x = x.copy().val.get_full_data()
        cond1 = where(logical_and(x > 0., x < .5))
        cond2 = where(x >= .5)
        not_cond = where(x <= 0.)
        x[cond2] -= 0.5
        x[cond2] **= 2
        x[cond1] = 0.
        x[not_cond] -= 1
        return Field(domain=dom, val=x)

    def derivative(self, x):
        dom = x.domain
        x = x.copy().val.get_full_data()
        cond1 = where(logical_and(x > 0., x < 0.5))
        cond2 = where(x > .5)
        not_cond = where(x <= 0.)
        x[cond2] -= 0.5
        x[cond2] *= 2
        x[cond1] = 0.
        x[not_cond] = 1
        return Field(domain=dom, val=x)
