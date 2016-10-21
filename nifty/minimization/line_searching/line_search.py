import abc

from keepers import Loggable

from ..bare_dot import bare_dot


class LineSearch(object, Loggable):
    """
    Class for finding a step size.◙
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):

        """
        Parameters
        ----------

        f : callable f(x, *args)
            Objective function.

        fprime : callable f'(x, *args)
            Objective functions gradient.

        f_args : tuple (optional)
            Additional arguments passed to objective function and its
            derivation.
        """

        self.xk = None
        self.pk = None

        self.f_k = None
        self.f_k_minus_1 = None
        self.fprime_k = None

    def set_functions(self, f, fprime, f_args=()):
        assert(callable(f))
        assert(callable(fprime))

        self.f = f
        self.fprime = fprime
        self.f_args = f_args

    def _set_coordinates(self, xk, pk, f_k=None, fprime_k=None,
                         f_k_minus_1=None):
        """
        Set the coordinates for a new line search.

        Parameters
        ----------
        xk : ndarray, d2o, field
            Starting point.

        pk : ndarray, d2o, field
            Unit vector in search direction.

        f_k : float (optional)
            Function value f(x_k).

        fprime_k : ndarray, d2o, field (optional)
            Function value fprime(xk).

        """

        self.xk = xk.copy()
        self.pk = pk.copy()

        if f_k is None:
            self.f_k = self.f(xk)
        else:
            self.f_k = f_k

        if fprime_k is None:
            self.fprime_k = self.fprime(xk)
        else:
            self.fprime_k = fprime_k

        if f_k_minus_1 is not None:
            f_k_minus_1 = f_k_minus_1.copy()
        self.f_k_minus_1 = f_k_minus_1

    def _phi(self, alpha):
        if alpha == 0:
            value = self.f_k
        else:
            value = self.f(self.xk + self.pk*alpha, *self.f_args)
        return value

    def _phiprime(self, alpha):
        if alpha == 0:
            gradient = self.fprime_k
        else:
            gradient = self.fprime(self.xk + self.pk*alpha, *self.f_args)

        return bare_dot(gradient, self.pk)

    @abc.abstractmethod
    def perform_line_search(self, xk, pk, f_k=None, fprime_k=None,
                            f_k_minus_1=None):
        raise NotImplementedError
