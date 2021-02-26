import numpy as np
from ..operators.multifield_flattener import MultifieldFlattener
from ..operators.simple_linear_operators import FieldAdapter
from ..operators.energy_operators import EnergyOperator
from ..operators.sandwich_operator import SandwichOperator
from ..sugar import full, from_random
from ..linearization import Linearization
from ..field import Field
from ..multi_field import MultiField

class MeanfieldModel():
    def __init__(self, domain):
        self.domain = domain
        self.Flat = MultifieldFlattener(self.domain)

        self.std = FieldAdapter(self.Flat.target,'var').absolute()
        self.latent = FieldAdapter(self.Flat.target,'latent')
        self.mean = FieldAdapter(self.Flat.target,'mean')
        self.generator = self.Flat.adjoint(self.mean + self.std * self.latent)
        self.entropy = GaussianEntropy(self.std.target) @ self.std

    def get_initial_pos(self, initial_mean=None):
        initial_pos = from_random(self.generator.domain).to_dict()
        initial_pos['latent'] = full(self.generator.domain['latent'], 0.)
        initial_pos['var'] = full(self.generator.domain['var'], 1.)

        if initial_mean is None:
            initial_mean = 0.1*from_random(self.generator.target)

        initial_pos['mean'] = self.Flat(initial_mean)
        return MultiField.from_dict(initial_pos)


class GaussianEntropy(EnergyOperator):
    def __init__(self, domain):
        self._domain = domain

    def apply(self, x):
        self._check_input(x)
        res =  -0.5* (2*np.pi*np.e*x**2).log().sum()
        if not isinstance(x, Linearization):
            return Field.scalar(res)
        if not x.want_metric:
            return res
        return res.add_metric(SandwichOperator.make(res.jac)) #FIXME not sure about metric