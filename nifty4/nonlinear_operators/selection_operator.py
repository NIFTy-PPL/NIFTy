from ..multi import MultiDomain, MultiField
from ..operators import LinearOperator
from ..sugar import full


class SelectionOperator(LinearOperator):
    def __init__(self, domain, key):
        if not isinstance(domain, MultiDomain):
            raise TypeError("Domain must be a MultiDomain")
        self._target = domain[key]
        self._domain = domain
        self._key = key

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return x[self._key].copy()
        else:
            result = {}
            for key, val in self.domain.items():
                if key != self._key:
                    result[key] = full(val, 0.)
                else:
                    result[key] = x.copy()
            return MultiField(result)
