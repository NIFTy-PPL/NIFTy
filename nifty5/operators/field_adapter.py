from __future__ import absolute_import, division, print_function

from ..compat import *
from .linear_operator import LinearOperator
from ..multi.multi_domain import MultiDomain
from ..multi.multi_field import MultiField


class FieldAdapter(LinearOperator):
    def __init__(self, dom, name_dom):
        self._domain = MultiDomain.make({name_dom: dom})
        self._target = dom

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self._all_ops

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            return x.values()[0]
        return MultiField(self._domain, (x,))
