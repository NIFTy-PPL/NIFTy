import numpy as np
from .space import Space


class DOFSpace(Space):
    def __init__(self, dof_weights):
        super(DOFSpace, self).__init__()
        self._dvol = tuple(dof_weights)
        self._needed_for_hash += ['_dvol']

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (len(self._dvol),)

    @property
    def dim(self):
        return len(self._dvol)

    def scalar_dvol(self):
        return None

    def dvol(self):
        return np.array(self._dvol)

    def __repr__(self):
        return 'this is a dof space'
