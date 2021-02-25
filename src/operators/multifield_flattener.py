

import numpy as np

from .linear_operator import LinearOperator
from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain
from ..sugar import makeField
from ..multi_field import MultiField

class MultifieldFlattener(LinearOperator):
    '''
    Flattens a MultiField and returns a Field in an unstructred domain with the same number of DOF.
    '''
    def __init__(self, domain):
        self._dof = domain.size
        self._domain = domain
        self._target = DomainTuple.make(UnstructuredDomain(self._dof))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def _flatten(self, x):
        result = np.empty(self.target.shape)
        runner = 0
        for key in self.domain.keys():
            dom_size = x[key].domain.size
            result[runner:runner+dom_size] = x[key].val.flatten()
            runner += dom_size
        return result

    def _restructure(self, x):
        runner = 0
        unstructured = x.val
        result = {}
        for key in self.domain.keys():
            subdom = self.domain[key]
            dom_size = subdom.size
            subfield = unstructured[runner:runner+dom_size].reshape(subdom.shape)
            subdict = {key:makeField(subdom,subfield)}
            result = {**result,**subdict}
            runner += dom_size
        return result

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            return makeField(self.target,self._flatten(x))
        return MultiField.from_dict(self._restructure(x))