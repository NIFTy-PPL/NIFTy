from . import NLTensor


class Variable(NLTensor):
    def __init__(self, domain):
        self._domain = domain

    def __call__(self, x):
        raise ValueError

    def eval(self, x):
        pass

    @property
    def derivative(self):
        pass
