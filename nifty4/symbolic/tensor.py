from ..operators.linear_operator import LinearOperator
from ..operators.scaling_operator import ScalingOperator
from ..operators.diagonal_operator import DiagonalOperator
from ..field import Field


class Tensor(object):
    def __init__(self, thing, rank, domain=None, name=None):
        """
        thing:   Can be a LinearOperator, Field or a Scalar.
        """
        assert isinstance(thing, LinearOperator) or isinstance(thing, Field) or isinstance(thing, (int, float))
        self.rank = rank
        self._name = name

        # Rank 2
        if self.rank == 2:
            if isinstance(thing, LinearOperator):
                self._thing = thing
            elif isinstance(thing, Field) and thing.domain == domain:
                self._thing = DiagonalOperator(thing)
            elif isinstance(thing, Field) and thing.domain != domain and len(thing.domain) == 0:
                assert domain is not None
                self._thing = ScalingOperator(thing[()], domain)
            elif isinstance(thing, (float, int)) and domain is not None:
                self._thing = ScalingOperator(thing, domain)
            else:
                raise ValueError

        # Rank 1
        elif self.rank == 1:
            if domain is not None and thing.domain != domain and len(thing.domain) == 0:
                thing = Field(domain, thing.val)
            elif isinstance(thing, Field):
                self._thing = thing
            else:
                raise ValueError

        # Rank 0
        elif self.rank == 0:
            if isinstance(thing, Field) and len(thing.domain) == 0:
                self._thing = thing
            elif isinstance(thing, (float, int)):
                self._thing = Field((), thing)
            else:
                raise ValueError

        else:
            raise NotImplementedError

    @property
    def domain(self):
        return self._thing.domain

    def __str__(self):
        if self._name is None:
            return str(self._thing)
        else:
            return self._name

    def __add__(self, other):
        assert self.rank == other.rank
        return self.__class__(self._thing + other._thing, self.rank)
