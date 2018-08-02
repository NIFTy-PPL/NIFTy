from __future__ import absolute_import, division, print_function

from ..compat import *
from .linear_operator import LinearOperator
from ..multi.multi_domain import MultiDomain
from ..multi.multi_field import MultiField


class FieldAdapter(LinearOperator):
    def __init__(self, dom, name_dom):
        self._domain = MultiDomain.make(dom)
        self._smalldom = MultiDomain.make({name_dom: self._domain[name_dom]})
        self._name = name_dom
        self._target = dom[name_dom]

    @property
    def capability(self):
        return self._all_ops

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            return x[self._name]
        tmp = MultiField(self._smalldom, (x,))
        return tmp.unite(MultiField.full(self._domain, 0.))
