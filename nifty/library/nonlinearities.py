from ..field import Field, exp, tanh


class Linear(object):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return Field.ones_like(x)


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


class PositiveTanh(object):
    def __call__(self, x):
        return 0.5 * tanh(x) + 0.5

    def derivative(self, x):
        return 0.5 * (1. - tanh(x)**2)
