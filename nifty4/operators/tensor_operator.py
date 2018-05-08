from .linear_operator import LinearOperator
from ..field import Field


class TensorOperator(object):
    """
    Supports only tensors of rank <= 2.
    """

    def __init__(self, indices, thing):
        """
        thing:   Can be a LinearOperator, Field or a Scalar.
        indices: Tuple of indices. -1 means covariant and +1 means
                 contravariant.
        """
        self._indices = indices
        self._thing = thing

    @property
    def rank(self):
        return len(self._indices)

    @property
    def indices(self):
        return self._indices

    def contract(self, op, index=0):
        assert op.rank == 1
        assert op.indices[0] == -self._indices[index]
        assert index in (0, 1)

        s = op._thing

        if isinstance(self._thing, LinearOperator):
            lop = self._thing
            if index == 1:
                # Ordinary linear operator application
                return self.__class__(self.indices[0:1], lop(s))
            # Linear operator application from the left
            return self.__class__(self.indices[1:2], lop.adjoint(s))

        if isinstance(self._thing, Field):
            # Scalar multiplication
            indices = ()
            return self.__class__(indices, self._thing.vdot(op._thing))



