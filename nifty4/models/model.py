import nifty4 as ift

from .selection_operator import SelectionOperator


class Model(object):
    def __init__(self, position):
        self._position = position

    def at(self, position):
        raise NotImplementedError

    @property
    def position(self):
        return self._position

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    def __getitem__(self, key):
        sel = SelectionOperator(self.value.domain, key)
        return sel(self)

    def __add__(self, other):
        assert isinstance(other, Model)
        return Add.make(self, other)

    def __sub__(self, other):
        assert isinstance(other, Model)
        return Add.make(self, (-1) * other)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return ScalarMul(self._position, other, self)
        if isinstance(other, Model):
            return Mul.make(self, other)
        raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return self.__mul__(other)
        raise NotImplementedError


def _joint_position(op1, op2):
    a = op1.position._val
    b = op2.position._val
    # Note: In python >3.5 one could do {**a, **b}
    ab = a.copy()
    ab.update(b)
    return ift.MultiField(ab)


class Mul(Model):
    """
    Please note: If you multiply two operators which share some keys in the
    position but have different values there, it is not guaranteed which value
    will be used for the product.
    """
    def __init__(self, position, op1, op2):
        super(Mul, self).__init__(position)

        self._op1 = op1.at(position)
        self._op2 = op2.at(position)

        self._value = self._op1.value * self._op2.value
        self._gradient = (ift.makeOp(self._op1.value) * self._op2.gradient +
                          ift.makeOp(self._op2.value) * self._op1.gradient)

    @staticmethod
    def make(op1, op2):
        position = _joint_position(op1, op2)
        return Mul(position, op1, op2)

    def at(self, position):
        return self.__class__(position, self._op1, self._op2)


class Add(Model):
    """
    Please note: If you add two operators which share some keys in the position
    but have different values there, it is not guaranteed which value will be
    used for the sum.
    """
    def __init__(self, position, op1, op2):
        super(Add, self).__init__(position)

        self._op1 = op1.at(position)
        self._op2 = op2.at(position)

        self._value = self._op1.value + self._op2.value
        self._gradient = self._op1.gradient + self._op2.gradient

    @staticmethod
    def make(op1, op2):
        position = _joint_position(op1, op2)
        return Add(position, op1, op2)

    def at(self, position):
        return self.__class__(position, self._op1, self._op2)


class ScalarMul(Model):
    def __init__(self, position, factor, op):
        super(ScalarMul, self).__init__(position)
        assert isinstance(factor, (float, int))
        assert isinstance(op, Model)

        self._op = op.at(position)
        self._factor = factor

        self._value = self._factor * self._op.value
        self._gradient = self._factor * self._op.gradient

    def at(self, position):
        return self.__class__(position, self._factor, self._op)


class LinearModel(Model):
    def __init__(self, inp, lin_op):
        """
        Computes lin_op(inp) where lin_op is a Linear Operator
        """
        from ..operators import LinearOperator
        super(LinearModel, self).__init__(inp.position)

        if not isinstance(lin_op, LinearOperator):
            raise TypeError("needs a LinearOperator as input")

        self._lin_op = lin_op
        self._inp = inp
        if isinstance(self._lin_op, SelectionOperator):
            self._lin_op = SelectionOperator(self._inp.value.domain,
                                             self._lin_op._key)

        self._value = self._lin_op(self._inp.value)
        self._gradient = self._lin_op*self._inp.gradient

    def at(self, position):
        return self.__class__(self._inp.at(position), self._lin_op)
