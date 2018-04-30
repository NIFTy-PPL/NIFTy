from ..operators.linear_operator import LinearOperator


class MultiLinearOperator(LinearOperator):
    @staticmethod
    def _toOperator(thing, dom):
        #from .multi_scaling_operator import ScalingOperator
        if isinstance(thing, MultiLinearOperator):
            return thing
        #if np.isscalar(thing):
        #    return MultiScalingOperator(thing, dom)
        return NotImplemented

    def __mul__(self, other):
        from .multi_chain_operator import MultiChainOperator
        other = self._toOperator(other, self.domain)
        return MultiChainOperator.make([self, other])

    def __rmul__(self, other):
        from .multi_chain_operator import MultiChainOperator
        other = self._toOperator(other, self.target)
        return MultiChainOperator.make([other, self])

    def __add__(self, other):
        from .multi_sum_operator import MultiSumOperator
        other = self._toOperator(other, self.domain)
        return MultiSumOperator.make([self, other], [False, False])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from .multi_sum_operator import MultiSumOperator
        other = self._toOperator(other, self.domain)
        return MultiSumOperator.make([self, other], [False, True])

    def __rsub__(self, other):
        from .multi_sum_operator import MultiSumOperator
        other = self._toOperator(other, self.domain)
        return MultiSumOperator.make([other, self], [False, True])
