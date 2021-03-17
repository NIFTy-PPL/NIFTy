from jax.numpy import sum, isscalar, array, abs
from jax.numpy.linalg import norm
from jax.tree_util import tree_unflatten


class Field():
    def __init__(self, domain, val):
        self.val, self.domain = val, domain

    def new(self, val):
        return Field(self.domain, val)

    def to_tree(self):
        return tree_unflatten(self.domain, self.val)

    def dot(self, other):
        if not isinstance(other, Field):
            raise TypeError("Can only perform dot product between fields")
        if other.domain != self.domain:
            raise ValueError("domains are incompatible.")
        res = [sum(v * w) for v, w in zip(self.val, other.val)]
        return sum(array(res))

    def squared_norm(self):
        res = [sum(v**2) for v in self.val]
        return sum(array(res))

    def norm(self, ord):
        my_norm = lambda x: abs(x) if (isscalar(x) or len(x.shape) == 0
                                      ) else norm(x, ord=ord)
        res = [my_norm(v) for v in self.val]
        return norm(array(res), ord=ord)

    def _unary_op(self, op):
        return self.new([getattr(v, op)() for v in self.val])

    def _binary_op(self, other, op):
        if isinstance(other, Field):
            # if other is a field, make sure that the domains match
            if other.domain != self.domain:
                raise ValueError("domains are incompatible.")
            it = iter(other.val)
        elif isscalar(other):
            from itertools import repeat
            it = repeat(other, len(self.val))
        else:
            raise ValueError(
                "Invalid binary op for Field and {}".format(type(other))
            )
        return self.new([getattr(v, op)(o) for v, o in zip(self.val, it)])

for op in ["__neg__", "__pos__", "__abs__"]:
    def func(op):
        def func2(self):
            return self._unary_op(op)

        return func2

    setattr(Field, op, func(op))

for op in [
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__", "__pow__",
    "__rpow__", "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"
]:

    def func(op):
        def func2(self, other):
            return self._binary_op(other, op)

        return func2

    setattr(Field, op, func(op))
