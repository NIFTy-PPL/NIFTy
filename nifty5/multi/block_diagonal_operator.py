import numpy as np
from ..operators.endomorphic_operator import EndomorphicOperator
from .multi_domain import MultiDomain
from .multi_field import MultiField


class BlockDiagonalOperator(EndomorphicOperator):
    def __init__(self, operators):
        """
        Parameters
        ----------
        operators : dict
            dictionary with operators domain names as keys and
            LinearOperators as items
        """
        super(BlockDiagonalOperator, self).__init__()
        self._operators = operators
        self._domain = MultiDomain.make(
            {key: op.domain for key, op in self._operators.items()})
        self._cap = self._all_ops
        for op in self._operators.values():
            self._cap &= op.capability

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return self._cap

    def apply(self, x, mode):
        self._check_input(x, mode)
        val = tuple(self._operators[key].apply(x._val[i], mode=mode)
                    for i, key in enumerate(x.keys()))
        return MultiField(self._domain, val)

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        dtype = MultiField.build_dtype(dtype, self._domain)
        val = tuple(self._operators[key].draw_sample(from_inverse, dtype[key])
                    for key in self._domain._keys)
        return MultiField(self._domain, val)

    def _combine_chain(self, op):
        res = {}
        for key in self._operators.keys():
            res[key] = self._operators[key]*op._operators[key]
        return BlockDiagonalOperator(res)

    def _combine_sum(self, op, selfneg, opneg):
        from ..operators.sum_operator import SumOperator
        res = {}
        for key in self._operators.keys():
            res[key] = SumOperator.make([self._operators[key],
                                         op._operators[key]],
                                        [selfneg, opneg])
        return BlockDiagonalOperator(res)
