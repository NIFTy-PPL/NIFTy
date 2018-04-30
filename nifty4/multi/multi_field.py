from ..field import Field
import numpy as np
from .multi_domain import MultiDomain


class MultiField(object):
    def __init__(self, val):
        """
        Parameters
        ----------
        val : dict
        """
        self._val = val

    def __getitem__(self, key):
        return self._val[key]

    def keys(self):
        return self._val.keys()

    def items(self):
        return self._val.items()

    def values(self):
        return self._val.values()

    @property
    def domain(self):
        return MultiDomain({key: val.domain for key, val in self._val.items()})

    def _check_domain(self, other):
        if other.domain != self.domain:
            raise ValueError("domains are incompatible.")

    def vdot(self, x):
        result = 0.
        self._check_domain(x)
        for key, sub_field in self.items():
            result += sub_field.vdot(x[key])
        return result

    def lock(self):
        for v in self.values():
            v.lock()
        return self

    def copy(self):
        return MultiField({key: val.copy() for key, val in self.items()})

    @staticmethod
    def zeros(domain, dtype=None):
        return MultiField({key: Field.zeros(dom, dtype=dtype)
                           for key, dom in domain.items()})

    @staticmethod
    def ones(domain, dtype=None):
        return MultiField({key: Field.ones(dom, dtype=dtype)
                           for key, dom in domain.items()})

    @staticmethod
    def empty(domain, dtype=None):
        return MultiField({key: Field.empty(dom, dtype=dtype)
                           for key, dom in domain.items()})

    def norm(self):
        """ Computes the L2-norm of the field values.

        Returns
        -------
        norm : float
            The L2-norm of the field values.
        """
        return np.sqrt(np.abs(self.vdot(x=self)))

    def __neg__(self):
        return MultiField({key: -val for key, val in self.items()})

    def conjugate(self):
        return MultiField({key: sub_field.conjugate()
                           for key, sub_field in self.items()})


for op in ["__add__", "__radd__", "__iadd__",
           "__sub__", "__rsub__", "__isub__",
           "__mul__", "__rmul__", "__imul__",
           "__div__", "__rdiv__", "__idiv__",
           "__truediv__", "__rtruediv__", "__itruediv__",
           "__floordiv__", "__rfloordiv__", "__ifloordiv__",
           "__pow__", "__rpow__", "__ipow__",
           "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"]:
    def func(op):
        def func2(self, other):
            if isinstance(other, MultiField):
                self._check_domain(other)
                result_val = {key: getattr(sub_field, op)(other[key])
                              for key, sub_field in self.items()}
            else:
                result_val = {key: getattr(val, op)(other)
                              for key, val in self.items()}
            return MultiField(result_val)
        return func2
    setattr(MultiField, op, func(op))
