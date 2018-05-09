from .linear_operator import LinearOperator
from .diagonal_operator import DiagonalOperator
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
        assert isinstance(thing, LinearOperator) or isinstance(thing, Field) or thing == 0
        assert isinstance(indices, (list, tuple))
        self._indices = indices
        if self.rank == 2 and isinstance(thing, Field):
            self._thing = DiagonalOperator(thing)
        else:
            self._thing = thing

    def __str__(self):
        return 'Tensor({})_{}'.format(self._thing, self._indices)

    def __add__(self, other):
        if isinstance(self, ZeroTensor):
            return other
        if isinstance(other, ZeroTensor):
            return self

        assert self.indices == other.indices
        return Tensor(self.indices, self._thing + other._thing)

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
        `op` needs to be a tensor of rank 1 or 2 or a ZeroTensor of rank 3.
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
                return self.__class__(indices, Field((), self._thing.vdot(op._thing)))
        if op.rank == 2:
            if isinstance(self._thing, LinearOperator):
                lop = self._thing
                # FIXME This is only a hack and does not fit all cases
                return self.__class__(self.indices[0:1] + op.indices[1:2], lop * op._thing)
            if isinstance(self._thing, Field):
                return self.__class__(op.indices[1:2], op._thing.adjoint(self._thing))

        if op.rank == 3 and isinstance(op, ZeroTensor):
            indices = self.indices[0:index] + self.indices[index:] + op.indices[1:]
            return ZeroTensor(indices)

        print(self)
        print(op)
        print('Contraction index: {}'.format(index))
        raise NotImplementedError


class ZeroTensor(Tensor):
    def __init__(self, indices):
        assert len(indices) <= 4
        self._indices = indices
        self._thing = 0

    def __str__(self):
        return '(Zero)_{}'.format(self._indices)

    def contract(self, op, index=0):
        """
        Supports rank <= 2 tensors as `op`.
        Takes always first index of op to contract with.
        """
        assert op.rank in (1, 2)
        assert op.indices[0] == -self._indices[index]
        assert index in (0, 1, 2)

        if self.rank == 1:
            return Tensor((), 0.)

        slc1 = slice(0, index)
        slc2 = slice(index + 1, None)
        indices = self.indices[slc1] + self.indices[slc2]

        if op.rank == 2:
            indices += op.indices[1:2]
        return self.__class__(indices)
