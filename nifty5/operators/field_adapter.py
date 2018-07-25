from __future__ import absolute_import, division, print_function

from ..compat import *
from .linear_operator import LinearOperator
from ..multi.multi_domain import MultiDomain
from ..multi.multi_field import MultiField


class FieldAdapter(LinearOperator):
    def __init__(self, op, name_dom, name_tgt):
        if name_dom is None:
            self._domain = op.domain
        else:
            self._domain = MultiDomain.make({name_dom: op.domain})
        if name_tgt is None:
            self._target = op.target
        else:
            self._target = MultiDomain.make({name_tgt: op.target})
        self._op = op

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        self._check_input(x, mode)

        dom = self._dom(mode)
        if x.domain is not dom:
            x = dom.values()[0]
        res = self._op.apply(x. mode)
        tgt = self._tgt(mode)
        if res.domain is not tgt:
            res = MultiField(tgt, (res,))
        return res
