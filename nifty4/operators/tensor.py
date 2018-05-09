from .linear_operator import LinearOperator
from ..field import Field


class Tensor(object):
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

    def __str__(self):
        return '({})_{}'.format(self._thing, self._indices)

    @property
    def rank(self):
        return len(self._indices)

    @property
    def indices(self):
        return self._indices

    @property
    def output(self):
        return self._thing

    def contract(self, op, index=0):
        """
        `op` needs to be a tensor of rank 1 or 2.
        `index` refers to the contracted index of `self`.
        """

        assert op.indices[0] == -self._indices[index]
        assert index in (0, 1)

        s = op._thing

        if op.rank == 1:
            if isinstance(self._thing, LinearOperator):
                lop = self._thing
                if index == 1:
                    # Ordinary linear operator application
                    t = lop(s)
                    return self.__class__(self.indices[0:1], t)
                # Linear operator application from the left
                return self.__class__(self.indices[1:2], lop.adjoint(s))

            if isinstance(self._thing, Field):
                # Scalar multiplication
                indices = ()
                return self.__class__(indices, self._thing.vdot(op._thing))
        if op.rank == 2:
            if isinstance(self._thing, LinearOperator):
                lop = self._thing
                # FIXME This is only a hack and does not fit all cases
                return self.__class__(self.indices[0] + op.indices[1], lop * op._thing)
            if isinstance(self._thing, Field):
                return self.__class__(op.indices[1], op._thing.adjoint(self._thing))

        raise NotImplementedError


class ZeroTensor(Tensor):
    def __init__(self, indices):
        assert len(indices) <= 3

    def contract(self, op, index=0):
        assert op.rank == 1
        assert op.indices[0] == -self._indices[index]
        assert index in (0, 1, 2)

        if self.rank == 1:
            return Tensor((), 0.)
        slc1 = slice(0, index)
        slc2 = slice(index, None)
        indices = self.indices[slc1] + self.indices[slc2]
        return self.__class__(indices)
